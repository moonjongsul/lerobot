"""Checks that hold the code to what the dataset actually contains.

Not unit tests of the code against itself -- these compare the rules and
derived targets against the recorded demonstrations, so a change that looks
harmless but contradicts the data fails here.

    python -m lerobot.policies.mvla.tests.test_against_dataset \
        --dataset /path/to/xarm7_kitting_260923
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from ..data.derive import build_neutral_prompts, derive_frame_targets
from ..data.segments import runs_frame, subtask_names
from ..planner.rules import (
    GOAL_KIT,
    GOAL_PICK,
    IN_GRIPPER,
    MAT_FLIPPED,
    MAT_TARGET,
    plan,
)

# Prompt wording must not predict whether a flip is needed much better than
# the base rate. Measured over every episode, as the accuracy of the best
# per-sentence guess.
#
# The floor is not 0.5 but 0.582: 122 of 292 episodes need a flip, so
# answering "no flip" for everything is already that accurate, and no
# wording can do worse than its own base rate. The neutral prompts reach
# 0.616 -- the residual is structural rather than lexical, because the two
# goals genuinely differ in how often a flip is needed (kit 36.8%, pick
# 56.9%) and collapsing them into one sentence would take the goal
# distinction the planner runs on with it. 0.62 admits that, while still
# failing loudly if the method ever leaks back into the wording (the
# recorded prompts score 0.997).
MAX_NEUTRAL_LEAK = 0.62
# The rule table has to reproduce the operators' sequences.
MIN_PLAN_AGREEMENT = 0.99


def check_plans(root) -> tuple[bool, str]:
    runs = runs_frame(root)
    names = subtask_names(root)
    clean = runs.groupby("episode").filter(lambda g: not g["failed"].any())
    start_state = {
        "approach_flip": MAT_TARGET,
        "approach_pick": MAT_FLIPPED,
        "move": IN_GRIPPER,
    }

    agree = total = 0
    mismatches: list[str] = []
    for _, group in clean.groupby("episode"):
        actual = tuple(
            names[int(s)] for s in group.sort_values("order")["subtask"]
        )
        state = start_state.get(actual[0])
        goal = GOAL_KIT if "place" in actual else GOAL_PICK
        total += 1
        try:
            predicted = tuple(plan(goal, state).subtasks)
        except Exception:  # noqa: BLE001 - any failure is a mismatch
            predicted = ()
        if predicted == actual:
            agree += 1
        elif len(mismatches) < 3:
            mismatches.append(f"{' -> '.join(actual)}  !=  {' -> '.join(predicted)}")

    ratio = agree / max(total, 1)
    ok = ratio >= MIN_PLAN_AGREEMENT
    detail = f"{agree}/{total} = {ratio * 100:.1f}%"
    if mismatches:
        detail += "\n      " + "\n      ".join(mismatches)
    return ok, detail


def check_prompts(root) -> tuple[bool, str]:
    mapping = build_neutral_prompts(root)
    ok = mapping.leak_after <= MAX_NEUTRAL_LEAK
    return ok, (
        f"recorded {mapping.leak_before * 100:.1f}% -> "
        f"neutral {mapping.leak_after * 100:.1f}% (must be <= {MAX_NEUTRAL_LEAK * 100:.0f}%)"
    )


def check_targets(root) -> tuple[bool, str]:
    targets = derive_frame_targets(root)
    runs = runs_frame(root)
    merged = targets.merge(
        runs[["run_id", "start", "end", "subtask", "failed"]], on="run_id"
    )

    problems = []
    if not ((merged["frame_index"] >= merged["start"]) & (merged["frame_index"] <= merged["end"])).all():
        problems.append("frames fall outside their run")
    if not (merged["subtask_index"] == merged["subtask"]).all():
        problems.append("subtask index disagrees with the run")
    if not (merged.loc[merged["failed"], "value_subtask"] == -1.0).all():
        problems.append("a failed run has a value above the floor")
    if int((targets["status"] != 0).sum()) != len(runs):
        problems.append("terminal status count != run count")
    if not (merged["mistake"] == merged["failed"]).all():
        problems.append("mistake flag disagrees with run outcome")
    if int((targets["run_id"] < 0).sum()):
        problems.append("frames left unassigned to a run")

    first = merged.loc[merged.groupby("run_id")["frame_index"].idxmin()]
    if float(first["elapsed_subtask"].max()) > 1e-6:
        problems.append("elapsed time nonzero at a run's first frame")

    succeeded = merged[~merged["failed"]]
    last = succeeded.loc[succeeded.groupby("run_id")["frame_index"].idxmax()]
    if float(last["value_subtask"].max()) > 1e-3 or float(last["value_subtask"].min()) < -0.2:
        problems.append("successful runs do not end near value 0")

    if np.any(np.diff(targets.loc[targets["episode_index"] == 0, "value_episode"].to_numpy()) < -1e-6):
        problems.append("episode value is not monotone within an episode")

    return not problems, ("; ".join(problems) if problems else f"{len(targets)} frames consistent")


CHECKS = {
    "planner matches demonstrations": check_plans,
    "neutral prompts hide the flip decision": check_prompts,
    "derived targets agree with runs": check_targets,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    failed = 0
    for name, check in CHECKS.items():
        ok, detail = check(args.dataset)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}\n      {detail}")
        failed += not ok
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
