"""Task planner: recognised state -> task prompt and subtask sequence.

Rules, not a learned model, because the mapping is nearly deterministic.
Checked against every failure-free episode in xarm7_kitting_260923, the
table below reproduces the operators' own subtask sequence 216 times out of
217 (99.5%):

    object needs flipping   approach_flip, flip, approach_pick, pick, move, place
    object ready to pick    approach_pick, pick, move, place
    object already held     move, place

The single mismatch is an episode that kits two parts in a row, which is a
missing outer loop rather than a wrong plan.

The hard part of this system is deciding *which* state the scene is in, not
what to do once you know. A learned planner would add a training loop and a
failure mode to a problem a lookup table answers, so the interface here is
shaped to be swapped for a VLM later while the body stays a table.

Prompts are the neutral ones: the recorded sentences name the method
("flip it, then pick it up"), which tells the policy the answer it is
supposed to read off the image.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..data.derive import NEUTRAL_PROMPTS

# Recognised object states the planner accepts.
MAT_TARGET = "target"  # on the mat, wrong face up
MAT_FLIPPED = "flipped"  # on the mat, ready to pick
IN_GRIPPER = "in_gripper"
UNKNOWN = "unknown_pose"

# Subtask short names, matching the editor's labels.
APPROACH_FLIP = "approach_flip"
FLIP = "flip"
APPROACH_PICK = "approach_pick"
PICK = "pick"
MOVE = "move"
PLACE = "place"

# Goals the planner can be asked to reach.
GOAL_KIT = "kit"
GOAL_PICK = "pick"
GOAL_PLACE = "place"

# Subtask sequence to reach each goal from each starting state. Read off the
# demonstrations rather than invented.
_PLANS: dict[tuple[str, str], list[str]] = {
    (GOAL_KIT, MAT_TARGET): [APPROACH_FLIP, FLIP, APPROACH_PICK, PICK, MOVE, PLACE],
    (GOAL_KIT, MAT_FLIPPED): [APPROACH_PICK, PICK, MOVE, PLACE],
    (GOAL_KIT, IN_GRIPPER): [MOVE, PLACE],
    (GOAL_PICK, MAT_TARGET): [APPROACH_FLIP, FLIP, APPROACH_PICK, PICK],
    (GOAL_PICK, MAT_FLIPPED): [APPROACH_PICK, PICK],
    (GOAL_PICK, IN_GRIPPER): [],
    (GOAL_PLACE, MAT_TARGET): [APPROACH_FLIP, FLIP, APPROACH_PICK, PICK, MOVE, PLACE],
    (GOAL_PLACE, MAT_FLIPPED): [APPROACH_PICK, PICK, MOVE, PLACE],
    (GOAL_PLACE, IN_GRIPPER): [MOVE, PLACE],
}

# Where to resume after a subtask fails. A failed manipulation is retried by
# re-approaching, never by repeating the manipulation from where it broke --
# which is what the operators did in all 92 recovered failures.
_RECOVERY: dict[str, str] = {
    FLIP: APPROACH_FLIP,
    PICK: APPROACH_PICK,
    MOVE: APPROACH_PICK,
    PLACE: APPROACH_PICK,
    APPROACH_FLIP: APPROACH_FLIP,
    APPROACH_PICK: APPROACH_PICK,
}


@dataclass
class Plan:
    goal: str
    state: str
    task_prompt: str
    subtasks: list[str] = field(default_factory=list)

    @property
    def done(self) -> bool:
        return not self.subtasks

    def __str__(self) -> str:
        return f"{self.task_prompt!r} via {' -> '.join(self.subtasks) or '(nothing to do)'}"


class UnplannableState(ValueError):
    """The recogniser did not report a state the planner can act on."""


def plan(goal: str, state: str) -> Plan:
    """Subtask sequence reaching `goal` from `state`.

    Raises on `unknown_pose` and on abstention rather than guessing: an
    object lying in neither canonical pose is not something the policy has
    demonstrations for, and guessing there is how a cell damages a part.
    """
    if goal not in {GOAL_KIT, GOAL_PICK, GOAL_PLACE}:
        raise UnplannableState(f"unknown goal {goal!r}")
    if state in (UNKNOWN, "abstain", "empty"):
        raise UnplannableState(
            f"state {state!r} has no plan: needs an operator, not a retry"
        )
    try:
        subtasks = list(_PLANS[(goal, state)])
    except KeyError as exc:
        raise UnplannableState(f"no plan for goal={goal!r} state={state!r}") from exc
    return Plan(goal=goal, state=state, task_prompt=NEUTRAL_PROMPTS[goal], subtasks=subtasks)


def replan_after_failure(current: Plan, failed_subtask: str) -> Plan:
    """Plan to resume with after `failed_subtask` did not achieve its goal.

    The remaining sequence is rebuilt from the recovery entry point rather
    than continued, so a failed place goes back through picking the part up
    again -- the recovery the demonstrations actually contain.
    """
    entry = _RECOVERY.get(failed_subtask)
    if entry is None:
        raise UnplannableState(f"no recovery defined for {failed_subtask!r}")
    full = _PLANS[(current.goal, MAT_TARGET if entry == APPROACH_FLIP else MAT_FLIPPED)]
    resume = full[full.index(entry) :]
    return Plan(
        goal=current.goal,
        state=current.state,
        task_prompt=current.task_prompt,
        subtasks=list(resume),
    )


def state_from_recognizer(mat_label: str, holding: bool) -> str:
    """Fuse the mat probe with the gripper signal into one state.

    Whether the part is held is read from the gripper, not from vision: the
    dataset has only 7 failed grasps, far too few to fit a classifier, while
    gripper width and effort separate "closed on something" from "closed on
    air" by construction.
    """
    if holding:
        return IN_GRIPPER
    if mat_label in (MAT_TARGET, MAT_FLIPPED, UNKNOWN):
        return mat_label
    return UNKNOWN if mat_label == "abstain" else mat_label
