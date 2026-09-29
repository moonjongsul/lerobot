"""30Hz 제어 루프의 핵심. ROS 없음 -- 노드와 replay가 같은 코드를 쓴다.

구조:

    tick(now) ── 30Hz, 절대 막히지 않는다
      ├─ worker.poll(now)      끝난 추론 결과 수거 (청크, 헤드)
      ├─ 상태 판단             정지 조건, 타임아웃, 플래너 결과
      ├─ worker.submit(...)    다음 청크가 곧 필요하면 / 헤드 읽을 때가 되면
      └─ executor.action_at    지금 시각에 해당하는 액션 -> 안전 제한 -> 명령

    worker ── 별도 스레드. 관측 -> 정책 -> 결과. 상태를 바꾸지 않는다.

모든 상태 변경은 tick 안에서만 일어난다. 그래서 잠금이 필요 없고, replay는
스레드 대신 가상 시계로 같은 순서를 재현할 수 있다.

**시간 기준 청크 실행.** 청크의 k번째 액션은 그 청크를 만든 관측 시각
`t_obs + k/fps`에 해당한다(데이터셋에서 프레임 t의 액션이 프레임 t의
관측과 같은 시각이듯이). 추론에 L초가 걸리면 새 청크는 인덱스 L*fps부터
쓰인다 -- 지연 보상이 따로 없이 저절로 된다. 다음 청크는 실행 지점
(`execute_steps`)보다 추론 지연만큼 앞서 요청하고, 도착하면 `blend_steps`
동안 이전 청크와 선형 혼합한다. 새 청크가 끝내 안 오면 이전 청크를
끝까지(50스텝) 쓰고, 그것도 다 쓰면 정지한다.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .action import ActionSafety, Command
from .config import RolloutConfig
from .observation import RawObservation, to_policy_observation
from .policy_runner import HeadRead, PolicyRunner

logger = logging.getLogger("ros_inference.controller")


# ─────────────────────────────────────────────────────────── chunk timing
class ChunkExecutor:
    def __init__(self, fps: float, execute_steps: int, blend_steps: int):
        self.fps = fps
        self.execute_steps = execute_steps
        self.blend_steps = blend_steps
        self.reset()

    def reset(self) -> None:
        self.chunk: np.ndarray | None = None
        self.t_obs = 0.0
        self._prev: tuple[np.ndarray, float] | None = None
        self._blend_start = 0.0

    def _index(self, now: float, t_obs: float) -> int:
        return int(math.floor((now - t_obs) * self.fps + 1e-6))

    def push(self, chunk: np.ndarray, t_obs: float, now: float) -> None:
        if self.chunk is not None and self.blend_steps > 0:
            self._prev = (self.chunk, self.t_obs)
            self._blend_start = now
        self.chunk, self.t_obs = chunk, t_obs

    def index(self, now: float) -> int:
        return self._index(now, self.t_obs) if self.chunk is not None else -1

    def action_at(self, now: float) -> np.ndarray | None:
        if self.chunk is None:
            return None
        i = max(0, self._index(now, self.t_obs))
        if i >= len(self.chunk):
            return None
        a = self.chunk[i]
        if self._prev is not None:
            w = (now - self._blend_start) * self.fps / self.blend_steps
            prev, prev_t = self._prev
            j = self._index(now, prev_t)
            if w < 1.0 and 0 <= j < len(prev):
                a = w * a + (1.0 - w) * prev[j]
            else:
                self._prev = None
        return a

    def needs_chunk(self, now: float, latency_s: float) -> bool:
        if self.chunk is None:
            return True
        lead = int(math.ceil(latency_s * self.fps))
        return self._index(now, self.t_obs) + lead >= self.execute_steps


# ───────────────────────────────────────────────────────────── inference
@dataclass
class Request:
    raw: RawObservation
    want_chunk: bool
    task: str = ""
    head_task: str | None = None
    elapsed: float | None = None
    submitted: float = 0.0


@dataclass
class Result:
    t_obs: float
    chunk: np.ndarray | None
    head: HeadRead | None
    task: str
    compute_s: float
    ready: float = 0.0


def compute(runner: PolicyRunner, cfg: RolloutConfig, req: Request) -> Result:
    t0 = time.perf_counter()
    obs = to_policy_observation(
        req.raw, runner.camera_names, tuple(cfg.ros.image_size), cfg.ros.image_rotate, cfg.ros.gripper_effort_max
    )
    head = runner.read_heads(obs, req.head_task, req.elapsed) if req.head_task is not None else None
    chunk = runner.infer_chunk(obs, req.task, req.elapsed) if req.want_chunk else None
    return Result(t_obs=req.raw.stamp, chunk=chunk, head=head, task=req.task,
                  compute_s=time.perf_counter() - t0)


class ThreadedWorker:
    """실기용. 요청 하나를 받아 백그라운드에서 계산한다."""

    def __init__(self, runner: PolicyRunner, cfg: RolloutConfig, clock: Callable[[], float] = time.monotonic):
        self.runner, self.cfg, self.clock = runner, cfg, clock
        self._in: queue.Queue[Request | None] = queue.Queue(maxsize=1)
        self._out: queue.Queue[Result] = queue.Queue()
        self._busy = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="inference", daemon=True)
        self._thread.start()

    @property
    def busy(self) -> bool:
        return self._busy.is_set()

    def submit(self, req: Request, now: float) -> bool:
        if self.busy:
            return False
        req.submitted = now
        self._busy.set()
        self._in.put(req)
        return True

    def poll(self, now: float) -> list[Result]:
        out = []
        while True:
            try:
                out.append(self._out.get_nowait())
            except queue.Empty:
                return out

    def _loop(self) -> None:
        while True:
            req = self._in.get()
            if req is None:
                return
            try:
                res = compute(self.runner, self.cfg, req)
                res.ready = self.clock()
                self._out.put(res)
            except Exception:  # noqa: BLE001 - 추론 실패가 제어 루프를 죽이면 안 된다
                logger.exception("inference failed")
            finally:
                self._busy.clear()

    def close(self) -> None:
        self._in.put(None)


class SimulatedWorker:
    """replay용. 즉시 계산하되, 결과는 가상 시계로 지연 후에 내준다.

    `latency_s`가 None이면 실제 계산 시간을 지연으로 쓴다.
    """

    def __init__(self, runner: PolicyRunner, cfg: RolloutConfig, latency_s: float | None = None):
        self.runner, self.cfg, self.latency_s = runner, cfg, latency_s
        self._pending: Result | None = None

    @property
    def busy(self) -> bool:
        return self._pending is not None

    def submit(self, req: Request, now: float) -> bool:
        if self.busy:
            return False
        res = compute(self.runner, self.cfg, req)
        res.ready = now + (res.compute_s if self.latency_s is None else self.latency_s)
        self._pending = res
        return True

    def poll(self, now: float) -> list[Result]:
        if self._pending is not None and now >= self._pending.ready:
            res, self._pending = self._pending, None
            return [res]
        return []

    def close(self) -> None:
        pass


# ──────────────────────────────────────────────────────────── controller
@dataclass
class TickOutput:
    command: Command | None       # None = 아무것도 보내지 않는다 (시작 전)
    status: str
    events: list[str] = field(default_factory=list)


class RolloutController:
    """상태: idle -> starting -> running -> (done | stopped | aborted)."""

    def __init__(self, cfg: RolloutConfig, runner: PolicyRunner, worker,
                 obs_source: Callable[[bool], RawObservation | None], recognizer=None):
        self.cfg = cfg
        self.runner = runner
        self.worker = worker
        self.recognizer = recognizer          # RecognizerWorker | SyncRecognizerWorker | None
        self.obs_source = obs_source
        self.executor = ChunkExecutor(cfg.control.fps, cfg.control.execute_steps, cfg.control.blend_steps)
        self.safety = ActionSafety(cfg.safety, cfg.control.fps)
        self.status = "idle"
        self.reason = ""
        self.tracker = None
        self.latency_s = 0.15
        self._t_start = 0.0
        self._last_head_req = -1e9
        self._start_mode = ""
        self.log: list[dict] = []
        self.last_head: HeadRead | None = None
        self.completion: dict | None = None   # 메인 prompt 작업 종료 결과

    # ─────────────────────────────────────────────────────── lifecycle
    def start(self, now: float, task: str | None = None) -> None:
        task = task or self.cfg.policy.task
        self.executor.reset()
        raw = self.obs_source(False)
        self.safety.reset(gripper_open=1.0 if raw is None or raw.gripper_position >= 0.5 else 0.0)
        self.log = []
        self.reason = ""
        self.completion = None
        self._t_start = now
        self.task = task
        self.tracker = None
        self.status = "starting"
        events: list[str] = []
        if self.runner.is_mvla:
            from .subtask import SubtaskTracker

            self.tracker = SubtaskTracker(task, self.runner.config, self.cfg.mvla, self.cfg.recognizer,
                                          recognizer_available=self.recognizer is not None)
            self._start_mode = self.tracker.start_mode()
            effort = raw.effort_normalized(self.cfg.ros.gripper_effort_max) if raw else 0.0
            if effort >= self.cfg.mvla.holding_effort or self._start_mode not in ("recognizer", "head"):
                # 쥐고 있거나 시작 상태가 고정이면 바로 plan
                events = self._handle(self.tracker.begin_from_head(None, effort, now))
            elif self._start_mode == "recognizer":
                self.tracker.schedule_start(now)
        logger.info(f"start: task={task!r} policy={'mvla' if self.runner.is_mvla else 'smolvla'} "
                    f"start_mode={self._start_mode or '-'} recognizer={self.recognizer is not None}")
        for e in events:
            logger.info(e)
        self._start_events = events

    def stop(self, reason: str = "operator") -> None:
        if self.status in ("starting", "running"):
            self.status = "stopped"
            self.reason = reason
            logger.info(f"stop: {reason}")

    @property
    def active(self) -> bool:
        return self.status in ("starting", "running")

    def _finish(self, status: str, reason: str) -> None:
        self.status, self.reason = status, reason
        logger.info(f"{status}: {reason}")

    def _complete(self, ev) -> None:
        """메인 prompt의 작업이 끝났다. 정지하고 결과를 남긴다."""
        self.completion = ev.data
        self._finish("done", f"task_complete: {ev.detail}")
        logger.info(f"task_complete {ev.data}")

    def snapshot(self, now: float) -> dict:
        """status 토픽과 로그에 쓰는 현재 상태."""
        snap = {
            "status": self.status,
            "reason": self.reason,
            "task": getattr(self, "task", None),
            "episode_elapsed_s": round(now - self._t_start, 2) if self.status != "idle" else 0.0,
            "latency_s": round(self.latency_s, 3),
            "task_complete": self.completion is not None,
        }
        if self.tracker is not None and self.tracker.started:
            snap.update(self.tracker.snapshot(now))
        if self.completion is not None:
            snap["completion"] = self.completion
        return snap

    # ──────────────────────────────────────────────────────────── tick
    def tick(self, now: float) -> TickOutput:
        events: list[str] = []
        if not self.active:
            for _ in self.worker.poll(now):
                pass  # 멈춘 뒤 도착한 결과는 버린다
            if self.recognizer is not None:
                self.recognizer.poll(now)
            return TickOutput(None if self.status == "idle" else self.safety.stop(), self.status)

        raw = self.obs_source(False)
        for res in self.worker.poll(now):
            self.latency_s = 0.8 * self.latency_s + 0.2 * max(0.0, res.ready - res.t_obs)
            if res.head is not None:
                events += self._on_head(res.head, res.t_obs)
            if res.chunk is not None and self.active:
                self.executor.push(res.chunk, res.t_obs, now)
        events += self._recognition(now, raw)

        if not self.active:  # 헤드·인식기 결과로 끝났을 수 있다
            return self._out(self.safety.stop(), events, raw, now)

        if raw is None or now - raw.stamp > self.cfg.control.stale_timeout_s:
            # 관측이 끊기면 멈추되 에피소드는 유지한다. 돌아오면 이어간다.
            return self._out(self.safety.stop(), events + ["stale_observation"], raw, now)
        if now - self._t_start > self.cfg.control.max_episode_s:
            self._finish("aborted", "episode timeout")
            return self._out(self.safety.stop(), events, raw, now)

        self._schedule(now, raw)

        action = self.executor.action_at(now) if self.status == "running" else None
        cmd = self.safety(action, raw.eef_position) if action is not None else self.safety.stop()
        return self._out(cmd, events, raw, now, action)

    def _recognition(self, now: float, raw: RawObservation | None) -> list[str]:
        """인식기: 판정 결과 반영, 릴리스 감시, 예약된 캡처 제출."""
        tr = self.tracker
        if self.recognizer is None or tr is None:
            return []
        effort = raw.effort_normalized(self.cfg.ros.gripper_effort_max) if raw else 0.0
        events: list[str] = []
        for v in self.recognizer.poll(now):
            events += self._handle(tr.on_verdict(v, effort, now))
        if raw is not None and self.active:
            tr.watch(raw.gripper_position, now)
            for purpose in tr.pop_due_captures(now):
                frame = self.obs_source(True)
                if frame is not None:
                    self.recognizer.submit(purpose, frame, now)
        return events

    def _schedule(self, now: float, raw: RawObservation) -> None:
        if self.worker.busy:
            return
        tr = self.tracker
        if self.status == "starting":
            if tr is None:
                self.status = "running"
            elif tr.started:
                self.status = "running"
            elif self._start_mode == "head":
                # 시작 상태를 정할 헤드 읽기 하나만. 청크는 plan이 선 뒤에.
                self._submit(now, want_chunk=False, head_task=tr.main_prompt(), elapsed=0.0)
            # recognizer: 예약된 시작 캡처의 판정을 기다린다
            return
        if self.executor.needs_chunk(now, self.latency_s):
            if tr is None:
                self._submit(now, want_chunk=True, task=self.task)
            else:
                el = tr.elapsed(now)
                self._submit(now, want_chunk=True, task=tr.policy_prompt(), head_task=tr.head_prompt(), elapsed=el)
            return
        if tr is not None and now - self._last_head_req >= 1.0 / max(self.cfg.mvla.head_hz, 1e-3):
            self._submit(now, want_chunk=False, head_task=tr.head_prompt(), elapsed=tr.elapsed(now))

    def _submit(self, now, *, want_chunk, task="", head_task=None, elapsed=None) -> None:
        raw = self.obs_source(True)
        if raw is None:
            return
        req = Request(raw=raw, want_chunk=want_chunk, task=task, head_task=head_task, elapsed=elapsed)
        if self.worker.submit(req, now) and head_task is not None:
            self._last_head_req = now

    def _handle(self, events) -> list[str]:
        """tracker 이벤트를 컨트롤러 상태로."""
        out = []
        for ev in events:
            out.append(f"{ev.kind}: {ev.detail}")
            if ev.kind == "begin" and self.status == "starting":
                self.status = "running"
            elif ev.kind == "task_complete":
                self._complete(ev)
            elif ev.kind == "needs_operator":
                self._finish("aborted", f"manager: {ev.detail}")
        return out

    def _on_head(self, head: HeadRead, t_obs: float) -> list[str]:
        self.last_head = head
        tr = self.tracker
        if tr is None:
            return []
        if not tr.started:
            if self._start_mode != "head":
                return []
            raw = self.obs_source(False)
            effort = raw.effort_normalized(self.cfg.ros.gripper_effort_max) if raw else 0.0
            return self._handle(tr.begin_from_head(head, effort, t_obs))
        return self._handle(tr.observe(head, t_obs))

    def _out(self, cmd: Command, events: list[str], raw, now: float, action=None) -> TickOutput:
        if getattr(self, "_start_events", None):
            events = self._start_events + events
            self._start_events = []
        for e in events:
            logger.info(e)
        h = self.last_head
        self.log.append({
            "t": now - self._t_start,
            "status": self.status,
            "subtask": self.tracker.subtask if self.tracker is not None and self.tracker.started else None,
            "chunk_index": self.executor.index(now),
            "latency_s": self.latency_s,
            "action": None if action is None else [float(x) for x in action],
            "command": [float(x) for x in cmd.twist],
            "gripper": cmd.gripper_target,
            "flags": list(cmd.flags),
            "remaining": self.tracker.remaining if self.tracker is not None and self.tracker.started else None,
            "value_subtask": None if h is None else h.value_subtask,
            "value_episode": None if h is None else h.value_episode,
            "head_subtask": None if h is None else h.subtask,
            "completion_streak": self.tracker.completion_streak if self.tracker is not None else None,
            "events": events,
        })
        return TickOutput(cmd, self.status, events)
