"""ROS 없이 rollout 파이프라인 전체를 녹화 에피소드로 돌려본다.

    python -m ros_inference.replay --config ros_inference/config/xarm7_kitting.yaml \
        --episode 33 [--latency 0.1] [--speed-scale 1.0]

실기 노드와 같은 `RolloutController`를 가상 시계로 돌린다. 관측은 데이터셋
프레임을 30Hz로 흘려 넣고, 이미지는 JPEG로 한 번 압축·복원해 CompressedImage
토픽을 흉내 낸다. 추론 지연은 실제 계산 시간(기본) 또는 `--latency`로 준다.

**개루프다.** 로봇이 명령대로 움직이지 않으므로 누적 오차는 재지 못한다.
재는 것은 파이프라인이 맞는지다: 단위·프레임·카메라 매핑·청크 타이밍·
안전 제한이 틀리면 명령과 녹화된 액션의 차이가 눈에 띄게 커진다. MVLA면
subtask 추적이 정답 경계를 따라가는지도 본다.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from .config import load_config
from .controller import RolloutController, SimulatedWorker
from .observation import RawObservation
from .policy_runner import PolicyRunner

logger = logging.getLogger("ros_inference.replay")


def load_episode(repo_id: str, episode: int, cameras: list[str], jpeg_quality: int = 90):
    """에피소드 프레임을 RawObservation 목록으로. 이미지는 JPEG 왕복."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, episodes=[episode], video_backend="torchcodec")
    frames, actions, subtasks, task = [], [], [], ""
    for i in range(len(ds)):
        item = ds[i]
        task = task or item["task"]
        images = {}
        for name in cameras:
            rgb = (item[f"observation.images.{name}"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            images[name] = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        state = item["observation.state"].numpy()
        pose = item["observation.state.eef_rot6d"].numpy()
        frames.append(RawObservation(
            stamp=float(item["frame_index"]) / ds.fps,
            images=images,
            eef_twist=state[:6].astype(np.float64),
            eef_position=pose[:3].astype(np.float64),
            gripper_position=float(state[6]),
            gripper_effort_raw=float(state[7]) * 600.0,  # 녹화값은 정규화돼 있다
        ))
        actions.append(item["action"].numpy())
        subtasks.append(int(item["subtask_index"]) if "subtask_index" in item else -1)
    return ds.fps, frames, np.asarray(actions), np.asarray(subtasks), task


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None, help="config의 체크포인트 대신")
    p.add_argument("--dataset", default="moonjongsul/xarm7-kitting-260923")
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--task", default=None, help="MVLA 목표 / SmolVLA 문장. 기본: 에피소드 과제에서")
    p.add_argument("--latency", type=float, default=None, help="가상 추론 지연(초). 기본은 실제 계산 시간")
    p.add_argument("--speed-scale", type=float, default=1.0,
                   help="녹화 액션과 비교하려면 1.0 (실기 설정의 0.5를 덮어쓴다)")
    p.add_argument("--recognizer", default=None,
                   help="인식기 디렉토리 (config 대신). 'none'이면 인식기 없이")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    cfg = load_config(args.config, overrides={"safety": {"speed_scale": args.speed_scale}})
    if args.checkpoint:
        cfg.policy.checkpoint = args.checkpoint
    runner = PolicyRunner(cfg.policy.checkpoint, cfg.policy.device, cfg.policy.num_inference_steps)
    fps, frames, rec_actions, gt_subtask, rec_task = load_episode(args.dataset, args.episode, runner.camera_names)
    if abs(fps - cfg.control.fps) > 1e-6:
        raise ValueError(f"dataset fps {fps} != control fps {cfg.control.fps}")

    task = args.task
    if task is None:
        if runner.is_mvla:
            from lerobot.policies.mvla.data.derive import TASK_GOALS

            task = TASK_GOALS[rec_task]
        else:
            task = rec_task

    cursor = {"i": 0}

    def obs_source(_with_images: bool):
        return frames[cursor["i"]]

    if args.recognizer is not None:
        cfg.recognizer.path = "" if args.recognizer == "none" else args.recognizer
    recog = None
    if runner.is_mvla and cfg.recognizer.path:
        from .recognition import Recognizer, SyncRecognizerWorker

        recog = SyncRecognizerWorker(Recognizer(cfg.recognizer.path, cfg.recognizer.device,
                                                tuple(cfg.ros.image_size), cfg.ros.image_rotate))

    worker = SimulatedWorker(runner, cfg, latency_s=args.latency)
    ctrl = RolloutController(cfg, runner, worker, obs_source, recognizer=recog)
    ctrl.cfg.control.max_episode_s = len(frames) / fps + 5.0
    ctrl.start(0.0, task)

    t_wall = time.time()
    cmds, acts = [], []
    for i in range(len(frames)):
        cursor["i"] = i
        out = ctrl.tick(i / fps)
        cmds.append(out.command.twist if out.command is not None else np.zeros(6))
        acts.append(ctrl.log[-1]["action"])
        if not ctrl.active:
            logger.info(f"controller ended at frame {i}: {ctrl.status} ({ctrl.reason})")
            break
    n = len(cmds)
    cmds = np.asarray(cmds)

    # 정책 액션(안전 제한 전)과 녹화 액션 비교. 첫 청크 전 구간은 뺀다.
    have = np.array([a is not None for a in acts])
    pol = np.array([a if a is not None else [np.nan] * 7 for a in acts], dtype=np.float64)
    rec = rec_actions[:n]
    names = ["vx", "vy", "vz", "wx", "wy", "wz", "gripper"]
    rmse = {k: float(np.sqrt(np.nanmean((pol[have, j] - rec[have, j]) ** 2))) for j, k in enumerate(names)}
    corr = {k: float(np.corrcoef(pol[have, j], rec[have, j])[0, 1]) for j, k in enumerate(names[:6])}
    summary = {
        "episode": args.episode, "task": task, "frames": n, "first_action_frame": int(np.argmax(have)),
        "status": ctrl.status, "reason": ctrl.reason,
        "mean_latency_s": float(np.mean([r["latency_s"] for r in ctrl.log])),
        "rmse_policy_vs_recorded": rmse, "corr_policy_vs_recorded": corr,
        "safety_flags": {f: int(sum(f in r["flags"] for r in ctrl.log))
                         for f in ("clip", "accel", "workspace", "stop", "invalid_action")},
        "wall_s": time.time() - t_wall,
    }
    if runner.is_mvla:
        from .subtask import SUBTASK_NAMES

        tracked = np.array([SUBTASK_NAMES.index(r["subtask"]) if r["subtask"] else -1 for r in ctrl.log])
        lab = gt_subtask[:n] >= 0
        summary["subtask_tracked_acc"] = float((tracked[lab] == gt_subtask[:n][lab]).mean())
        summary["events"] = [(r["t"], e) for r in ctrl.log for e in r["events"]]
        summary["completion"] = ctrl.completion

    out = Path(args.out or f"replay_ep{args.episode}.json")
    out.write_text(json.dumps({"summary": summary, "log": ctrl.log,
                               "recorded_action": rec.tolist(), "recorded_subtask": gt_subtask[:n].tolist()},
                              indent=1, default=float))
    print(json.dumps(summary, indent=2, default=float))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
