"""MVLA tracker: 인식기 판정·헤드 읽기에 따른 plan 진행. 모델 없이."""

import numpy as np

from lerobot.policies.mvla.configuration_mvla import MVLAConfig
from ros_inference.config import MVLARuntimeConfig, RecognizerRuntimeConfig
from ros_inference.policy_runner import HeadRead
from ros_inference.recognition import Verdict
from ros_inference.subtask import SUBTASK_NAMES, SubtaskTracker


def _tracker(goal="kit"):
    return SubtaskTracker(goal, MVLAConfig(), MVLARuntimeConfig(), RecognizerRuntimeConfig(),
                          recognizer_available=True)


def _mat(label):
    probs = {"empty": 0.0, "target": 0.0, "flipped": 0.0, "unknown_pose": 0.0}
    probs[label] = 1.0
    return {"label": label, "probs": probs, "ood": 1.0, "detected": True}


def _tray(bad):
    return {"label": "bad" if bad > 0.5 else "ok", "probs": {"ok": 1 - bad, "bad": bad}, "ood": 1.0,
            "detected": True}


def _head(subtask, v_sub=-0.5, v_ep=-0.5):
    p = np.zeros(7)
    p[SUBTASK_NAMES.index(subtask)] = 1.0
    return HeadRead(p, v_sub, 0.1, v_ep, np.array([1.0, 0.0, 0.0]))


def _kinds(events):
    return [e.kind for e in events]


def test_start_from_recognizer_builds_flip_plan():
    tr = _tracker()
    tr.schedule_start(0.0)
    assert tr.pop_due_captures(1.0) == ["start"] * 3
    ev = []
    for _ in range(3):
        ev += tr.on_verdict(Verdict("start", 0.1, _mat("target")), effort=0.0, t=0.8)
    assert "begin" in _kinds(ev)
    assert tr.remaining == ["approach_flip", "flip", "approach_pick", "pick", "move", "place"]


def test_flip_verification_failure_replans():
    tr = _tracker()
    tr.begin("target", 0.0)
    tr.observe(_head("flip"), 1.0)
    tr.observe(_head("flip"), 1.2)          # approach_flip -> flip
    tr.observe(_head("approach_pick"), 2.0)
    tr.observe(_head("approach_pick"), 2.2)  # flip -> approach_pick, flip 검증 예약
    assert tr.subtask == "approach_pick"
    assert tr.pop_due_captures(3.0) == ["flip"] * 3
    ev = []
    for _ in range(2):
        ev += tr.on_verdict(Verdict("flip", 2.3, _mat("target")), effort=0.0, t=2.5)
    assert "replan" in _kinds(ev)
    assert tr.subtask == "approach_flip"
    assert any(r.name == "flip" and r.outcome == "failed_verification" for r in tr.history)


def test_place_bad_then_tray_retry_then_complete():
    tr = _tracker()
    tr.begin("in_gripper", 0.0)
    tr.observe(_head("place"), 1.0)
    tr.observe(_head("place"), 1.2)         # move -> place
    assert tr.subtask == "place"
    tr.watch(0.0, 2.0)
    tr.watch(0.5, 2.1)                       # 그리퍼 열림 -> 릴리스
    assert tr.released_at == 2.1
    assert tr.pop_due_captures(5.0) == ["place"] * 3
    ev = []
    for _ in range(2):
        ev += tr.on_verdict(Verdict("place", 2.5, _tray(0.9)), effort=0.0, t=3.0)
    assert "replan" in _kinds(ev)
    assert tr.remaining == ["approach_pick", "pick", "place"]   # move 없이 트레이에서 재시도

    tr.observe(_head("pick"), 4.0)
    tr.observe(_head("pick"), 4.2)
    tr.observe(_head("place"), 5.0)
    tr.observe(_head("place"), 5.2)
    assert tr.subtask == "place"
    tr.watch(0.0, 6.0)
    tr.watch(0.6, 6.1)
    tr.pop_due_captures(9.0)
    ev = []
    for _ in range(2):
        ev += tr.on_verdict(Verdict("place", 6.5, _tray(0.05)), effort=0.0, t=7.0)
    done = [e for e in ev if e.kind == "task_complete"]
    assert done and done[0].data["basis"] == "recognizer_tray"
    assert done[0].data["retries"] == {"place": 1}


def test_unverifiable_last_subtask_completes_on_value():
    tr = _tracker(goal="pick")
    tr.begin("flipped", 0.0)
    tr.observe(_head("pick"), 1.0)
    tr.observe(_head("pick"), 1.2)          # approach_pick -> pick (마지막, 검증 불가)
    ev = []
    for k in range(3):
        ev += tr.observe(_head("pick", v_sub=-0.02), 2.0 + 0.2 * k)
    done = [e for e in ev if e.kind == "task_complete"]
    assert done and done[0].data["basis"] == "last_subtask_value"


def test_place_without_release_waits_then_completes_unverified():
    tr = _tracker()
    tr.begin("in_gripper", 0.0)
    tr.observe(_head("place"), 1.0)
    tr.observe(_head("place"), 1.2)
    ev = []
    for k in range(40):                       # 릴리스 없이 value만 높음
        ev += tr.observe(_head("place", v_sub=-0.01), 2.0 + 0.2 * k)
    done = [e for e in ev if e.kind == "task_complete"]
    assert done and done[0].data["basis"] == "unverified"
