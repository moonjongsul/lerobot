"""MVLA 런타임: 메인 prompt(목표) -> plan -> subtask 추적 -> 검증 -> 작업 종료.

    메인 prompt ──> 플래너 ──> plan [s1, ..., sN]
                                  │
    헤드(subtask, value) ─────> SubtaskTracker ──> 프롬프트 (Task + 현재 Subtask) ──> 정책
    인식기(mat, tray)   ─────>       │
                                  └── plan의 마지막 subtask 완료 = 메인 prompt 작업 종료

세 개의 시계 (설계 문서 §5.3):
  * 정책 30Hz
  * 헤드 ~5Hz -- subtask 전진, value 정체 감시, 검증 불가 subtask의 완료
  * 인식기, 경계에서만 -- 시작 상태, flip 결과, place 결과. 재시도/완료를 확정한다

규칙:
  * 시작: 그리퍼 effort로 파지 여부. 아니면 인식기 mat 판정(0.1/0.4/0.7초 3프레임)
    -> 플래너. 인식기가 없으면 헤드의 approach_flip/approach_pick 확률로 대체.
  * 전진: 헤드가 plan의 다음 subtask를 `hysteresis`번 연속 예측.
  * flip 검증: flip에서 넘어간 직후 +3/12/21프레임에 mat 판정. `flipped`가 아니면
    (2회 연속) approach_flip부터 재계획. 판정 동안 정책은 이미 approach_pick
    프롬프트로 움직인다 -- flip 프롬프트에 붙잡아 두면 다시 뒤집으려 할 수 있다.
  * place 검증: 그리퍼가 실제로 열린 순간(관측 위치 상승)을 릴리스로 보고
    +8/18/28프레임에 tray 판정. ok 2회 -> 완료, bad 2회 -> approach_pick부터 재시도.
  * 검증할 수 없는 마지막 subtask(pick 목표의 pick, 인식기 없음): value로 완료
    (`value_subtask` 또는 `value_episode` >= 임계가 `completion_consecutive`회).
  * place인데 릴리스가 끝내 안 보이면: value 완료 조건이 `unverified_timeout_s`
    동안 유지될 때 검증 없이 완료하고 근거를 `unverified`로 남긴다.
  * 실패: 매니저의 정체 규칙 또는 검증 실패 -> 재계획. 재시도 상한 초과 -> 정지.

**작업 종료를 plan으로 판단하는 이유:** 헤드의 `value_episode`는 끝 시점을
잘 맞히지만 목표 문장을 바꿔도 값이 그대로다(`eval_termination`). 목표에 따라
끝나는 지점이 바뀐다는 걸 보장하는 경로는 plan뿐이다.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

from lerobot.policies.mvla.data.derive import NEUTRAL_PROMPTS
from lerobot.policies.mvla.planner.manager import VERIFIED_BY, ManagerConfig, Outcome, TaskManager
from lerobot.policies.mvla.planner.rules import (
    APPROACH_FLIP, APPROACH_PICK, FLIP, IN_GRIPPER, MAT_FLIPPED, MAT_TARGET, PLACE,
)
from lerobot.policies.mvla.processor_mvla import PromptBuilder

from .config import MVLARuntimeConfig, RecognizerRuntimeConfig
from .policy_runner import HeadRead
from .recognition import Verdict, aggregate

logger = logging.getLogger("ros_inference.subtask")

# 헤드 클래스 번호 -> 짧은 이름. 데이터셋 meta/subtasks의 순서와 같다.
SUBTASK_NAMES = ["approach_flip", "flip", "approach_pick", "pick", "move", "place"]
# 짧은 이름 -> 학습 때 프롬프트에 들어간 문장 (meta/subtasks.parquet)
SUBTASK_SENTENCES = {
    "approach_flip": "approach to black plastic object for flip",
    "flip": "flip black plastic object upside down",
    "approach_pick": "approach to black plastic object for pick up",
    "pick": "pick up black plastic object",
    "move": "move black plastic object to white tray",
    "place": "place black plastic object in white tray",
}
FPS = 30.0


@dataclass
class TrackerEvent:
    kind: str       # begin | advance | replan | task_complete | needs_operator | verdict
    detail: str
    data: dict = field(default_factory=dict)


@dataclass
class SubtaskRecord:
    name: str
    t_start: float
    t_end: float | None = None
    outcome: str = "running"   # running | done | failed | failed_verification
    verdict: dict | None = None


class SubtaskTracker:
    def __init__(self, goal: str, policy_config, cfg: MVLARuntimeConfig,
                 recog: RecognizerRuntimeConfig | None = None, recognizer_available: bool = False):
        if goal not in NEUTRAL_PROMPTS:
            raise ValueError(f"MVLA task must be a goal name {sorted(NEUTRAL_PROMPTS)}, got {goal!r}")
        self.goal = goal
        self.cfg = cfg
        self.rcfg = recog or RecognizerRuntimeConfig()
        self.recognizer = recognizer_available
        self.builder = PromptBuilder(policy_config)
        self.manager = TaskManager(ManagerConfig(goal=goal, tray_bad_threshold=self.rcfg.tray_bad_threshold))
        self.neutral = NEUTRAL_PROMPTS[goal]
        self.t0 = 0.0
        self.t_begin = 0.0
        self.history: list[SubtaskRecord] = []
        self.initial_state: str | None = None
        self.start_verdict: dict | None = None
        self.completion_streak = 0
        self.last_head: HeadRead | None = None
        self._last_value_t: float | None = None
        self._pending = ("", 0)
        self._current: str | None = None
        self._captures: list[tuple[float, str]] = []
        self._frames: dict[str, list[dict]] = {"start": [], "flip": [], "place": []}
        self._grip_min: float | None = None
        self.released_at: float | None = None
        self._unverified_since: float | None = None
        self._retries_seen = 0

    # ────────────────────────────────────────────────────────── queries
    @property
    def started(self) -> bool:
        return self.manager.state is not None

    @property
    def subtask(self) -> str | None:
        return self.manager.state.subtask if self.manager.state else None

    @property
    def remaining(self) -> list[str]:
        return list(self.manager.state.plan.subtasks) if self.manager.state else []

    @property
    def outcome(self) -> Outcome:
        return self.manager.state.outcome if self.manager.state else Outcome.RUNNING

    def elapsed(self, t: float) -> float:
        return max(0.0, t - self.t0)

    def main_prompt(self) -> str:
        return self.builder.build_unconditional(self.neutral)

    def policy_prompt(self) -> str:
        sentence = SUBTASK_SENTENCES.get(self.subtask or "")
        return self.builder.build_inference(self.neutral, subtask=sentence, speed_bin_value=self.cfg.speed_bin)

    def head_prompt(self) -> str:
        return self.policy_prompt() if self.cfg.head_prompt == "self" else self.main_prompt()

    def _verifies(self, subtask: str) -> bool:
        if not self.recognizer or subtask not in VERIFIED_BY:
            return False
        return self.rcfg.verify_place if subtask == PLACE else self.rcfg.verify_flip

    # ───────────────────────────────────────────────────────────── start
    def start_mode(self) -> str:
        """recognizer | head | 고정 상태 이름."""
        mode = self.cfg.initial_state
        if mode == "auto":
            return "recognizer" if self.recognizer else "head"
        if mode == "recognizer" and not self.recognizer:
            raise RuntimeError("mvla.initial_state=recognizer but no recognizer is configured")
        return mode

    def schedule_start(self, t: float) -> None:
        self._captures += [(t + s, "start") for s in self.rcfg.start_offsets_s]

    def begin_from_head(self, head: HeadRead | None, effort: float, t: float) -> list[TrackerEvent]:
        mode = self.start_mode()
        if effort >= self.cfg.holding_effort:
            return self.begin(IN_GRIPPER, t)
        if mode in (MAT_TARGET, MAT_FLIPPED, IN_GRIPPER):
            return self.begin(mode, t)
        if head is None:
            raise RuntimeError("head-based start needs a head read")
        p = head.subtask_probs
        i_flip, i_pick = SUBTASK_NAMES.index(APPROACH_FLIP), SUBTASK_NAMES.index(APPROACH_PICK)
        return self.begin(MAT_TARGET if p[i_flip] >= p[i_pick] else MAT_FLIPPED, t, basis="head")

    def begin(self, mat_label: str, t: float, holding: bool = False, basis: str = "fixed") -> list[TrackerEvent]:
        holding = holding or mat_label == IN_GRIPPER
        st = self.manager.begin(mat_label, holding=holding)
        self.initial_state = st.plan.state
        self.t0 = self.t_begin = t
        self._current = st.subtask
        if st.subtask:
            self.history.append(SubtaskRecord(st.subtask, t))
        logger.info(f"plan: {st.plan}  (start {mat_label}, holding={holding}, basis={basis})")
        if st.outcome is Outcome.NEEDS_OPERATOR:
            return [TrackerEvent("needs_operator", st.reason)]
        if st.outcome is Outcome.DONE:
            return [TrackerEvent("task_complete", "goal already satisfied at start",
                                 self.completion_report(t, "already_satisfied"))]
        return [TrackerEvent("begin", f"{self.initial_state}: {st.plan}", {"basis": basis})]

    # ────────────────────────────────────────────────── recognizer timing
    def watch(self, gripper_position: float, t: float) -> None:
        """매 틱. place 중 그리퍼가 실제로 열리는 순간을 찾는다.

        학습 샘플의 릴리스는 관측 그리퍼 채널의 가장 큰 상승이었다. 런타임은
        미래를 모르므로 place 중 최저값에서 `release_rise` 이상 오른 첫 순간으로 본다.
        """
        if self.subtask != PLACE or not self._verifies(PLACE) or self.released_at is not None:
            return
        g = float(gripper_position)
        self._grip_min = g if self._grip_min is None else min(self._grip_min, g)
        if g - self._grip_min >= self.rcfg.release_rise:
            self.released_at = t
            self._captures += [(t + f / FPS, "place") for f in self.rcfg.release_offsets_frames]
            logger.info(f"release detected at {t:.2f}s (gripper {self._grip_min:.2f} -> {g:.2f})")

    def pop_due_captures(self, t: float) -> list[str]:
        due = [p for when, p in self._captures if when <= t]
        self._captures = [(when, p) for when, p in self._captures if when > t]
        return due

    # ──────────────────────────────────────────────────────── verdicts
    def on_verdict(self, v: Verdict, effort: float, t: float) -> list[TrackerEvent]:
        res = v.result
        events = [TrackerEvent("verdict", f"{v.purpose}: {res['label']}", {"purpose": v.purpose, **res})]
        if v.purpose == "start":
            self._frames["start"].append(res)
            if len(self._frames["start"]) < len(self.rcfg.start_offsets_s) or self.started:
                return events
            agg = aggregate(self._frames["start"])
            self.start_verdict = agg
            return events + self.begin(agg["label"], t, holding=effort >= self.cfg.holding_effort,
                                       basis="recognizer")

        st = self.manager.state
        if st is None or st.outcome is not Outcome.RUNNING:
            return events

        if v.purpose == "flip":
            self._frames["flip"].append(res)
            before = self.subtask
            self.manager.verification_failed(FLIP, res)
            if self.subtask != before or self.manager.state.outcome is not Outcome.RUNNING:
                for rec in reversed(self.history):
                    if rec.name == FLIP:
                        rec.outcome, rec.verdict = "failed_verification", aggregate(self._frames["flip"])
                        break
            return events + self._sync(t, basis="recognizer_mat")

        if v.purpose == "place":
            if self.subtask != PLACE:
                return events  # 재계획 뒤 늦게 도착한 판정
            self._frames["place"].append(res)
            self.manager.subtask_finished(verdict=res)
            if self.subtask == PLACE and self.manager.state.outcome is Outcome.RUNNING \
                    and len(self._frames["place"]) >= len(self.rcfg.release_offsets_frames):
                # 프레임끼리 엇갈려 히스테리시스를 못 채웠다: 평균 판정으로 결정
                agg = aggregate(self._frames["place"])
                self.manager.subtask_finished(verdict=agg)
                self.manager.subtask_finished(verdict=agg)
            if self.history and self.history[-1].name == PLACE and self.subtask != PLACE:
                self.history[-1].verdict = aggregate(self._frames["place"])
            return events + self._sync(t, basis="recognizer_tray")
        return events

    # ─────────────────────────────────────────────────────────── heads
    def observe(self, head: HeadRead, t: float) -> list[TrackerEvent]:
        """헤드 한 번 읽을 때마다."""
        self.last_head = head
        st = self.manager.state
        if st is None or st.outcome is not Outcome.RUNNING:
            return []

        dt = 0.0 if self._last_value_t is None else max(0.0, t - self._last_value_t)
        self._last_value_t = t
        self.manager.observe_value(head.value_subtask, dt)
        if self.subtask != self._current or self.manager.state.outcome is not Outcome.RUNNING:
            return self._sync(t, basis="stall")

        plan = st.plan.subtasks
        basis = "head"
        if len(plan) == 1:
            cfg = self.cfg
            hit = (head.value_subtask >= cfg.completion_value_subtask
                   or head.value_episode >= cfg.completion_value_episode)
            self.completion_streak = self.completion_streak + 1 if hit else 0
            if self.completion_streak >= cfg.completion_consecutive:
                if not self._verifies(plan[0]):
                    self.manager.subtask_finished(verdict=None)
                    basis = "last_subtask_value"
                elif self.released_at is None:
                    # 정책은 끝났다고 보는데 그리퍼가 안 열렸다. 잠시 기다린 뒤 검증 없이 완료.
                    self._unverified_since = self._unverified_since or t
                    if t - self._unverified_since >= self.rcfg.unverified_timeout_s:
                        logger.warning(f"{plan[0]}: no release seen; completing unverified")
                        self.manager.subtask_finished(verdict=None)
                        basis = "unverified"
            else:
                self._unverified_since = None
        else:
            self.completion_streak = 0
            nxt = plan[1]
            pred = SUBTASK_NAMES[head.subtask]
            if pred == nxt:
                last, count = self._pending
                count = count + 1 if last == pred else 1
                self._pending = (pred, count)
                if count >= self.cfg.hysteresis:
                    self._pending = ("", 0)
                    self.manager.subtask_finished(verdict=None)
            else:
                self._pending = ("", 0)
        return self._sync(t, basis=basis)

    # ─────────────────────────────────────────────────── state changes
    def _sync(self, t: float, basis: str) -> list[TrackerEvent]:
        """매니저 상태가 바뀌었으면 기록·타이머·검증 예약을 맞추고 이벤트를 낸다."""
        events: list[TrackerEvent] = []
        st = self.manager.state
        prev, new = self._current, st.subtask
        # 같은 subtask로 재계획되는 경우도 있다(approach_pick 정체 -> approach_pick).
        retried = st.total_retries != self._retries_seen
        self._retries_seen = st.total_retries
        if new != prev or retried:
            # 매니저는 실패(_fail)일 때만 재시도 수를 올린다. 그 외의 변화는 전진이다
            # -- plan 순서는 고정이 아니다(트레이 재시도는 pick -> place).
            advanced = not retried
            kind = "advance" if advanced else "replan"
            if self.history and self.history[-1].t_end is None:
                rec = self.history[-1]
                rec.t_end = t
                if rec.outcome == "running":
                    rec.outcome = "done" if kind == "advance" else "failed"
            if new:
                self.history.append(SubtaskRecord(new, t))
                detail = f"{prev} -> {new}" + (f" ({st.reason})" if kind == "replan" else "")
                events.append(TrackerEvent(kind, detail, {"basis": basis}))
            self._current = new
            self.t0 = t
            self.completion_streak = 0
            self._pending = ("", 0)
            self._grip_min, self.released_at, self._unverified_since = None, None, None
            self._frames["place"] = []
            self._captures = [(w, p) for w, p in self._captures if p == "start"]
            if kind == "advance" and prev == FLIP and self._verifies(FLIP):
                self._frames["flip"] = []
                self._captures += [(t + f / FPS, "flip") for f in self.rcfg.flip_offsets_frames]
        if st.outcome is Outcome.DONE:
            events.append(TrackerEvent("task_complete", f"goal '{self.goal}' reached",
                                       self.completion_report(t, basis)))
        elif st.outcome is Outcome.NEEDS_OPERATOR:
            events.append(TrackerEvent("needs_operator", st.reason))
        return events

    # ────────────────────────────────────────────────────────── report
    def completion_report(self, t: float, basis: str) -> dict:
        h = self.last_head
        return {
            "goal": self.goal,
            "main_prompt": self.main_prompt(),
            "initial_state": self.initial_state,
            "start_verdict": self.start_verdict,
            "basis": basis,
            "duration_s": round(t - self.t_begin, 2),
            "subtasks": [asdict(r) for r in self.history],
            "retries": dict(self.manager.state.retries) if self.manager.state else {},
            "value_subtask": None if h is None else round(h.value_subtask, 3),
            "value_episode": None if h is None else round(h.value_episode, 3),
        }

    def snapshot(self, t: float) -> dict:
        """status 토픽용 현재 상태."""
        h = self.last_head
        st = self.manager.state
        return {
            "goal": self.goal,
            "main_prompt": self.main_prompt(),
            "subtask": self.subtask,
            "remaining": self.remaining,
            "on_last_subtask": len(self.remaining) == 1,
            "subtask_elapsed_s": round(self.elapsed(t), 2),
            "task_elapsed_s": round(t - self.t_begin, 2),
            "completion_streak": self.completion_streak,
            "released": self.released_at is not None,
            "pending_verification": sorted({p for _, p in self._captures}),
            "retries": dict(st.retries) if st else {},
            "value_subtask": None if h is None else round(h.value_subtask, 3),
            "value_subtask_spread": None if h is None else round(h.value_subtask_spread, 3),
            "value_episode": None if h is None else round(h.value_episode, 3),
            "head_subtask": None if h is None else SUBTASK_NAMES[h.subtask],
        }
