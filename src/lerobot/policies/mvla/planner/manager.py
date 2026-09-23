"""Task manager: drives the plan, verifies subtasks, and re-prompts on failure.

Three clocks, deliberately separate, because they have different costs:

    30 Hz    the policy acts.
    30 Hz    the value head watches for stall -- cheap, it is already part
             of the forward pass.
    ~0.3 Hz  the recognizer renders a verdict at subtask boundaries. It is
             a separate model on full-resolution crops, so it runs when a
             subtask ends, not continuously.

The split matters for `place`: the value head can say "this is going badly"
early and continuously, but only the recognizer can say "that part is
sitting on the rib, not in the pocket", and that verdict is what gates a
retry. Gating on the fast signal alone trades a 0.98-AUC judgement for a
noisy one.

Retries are bounded. A recognizer that misreads a good placement as bad
would otherwise re-pick the same part forever, and each cycle is a chance
to damage it. On exhaustion the manager stops and asks for a human, which
is also what it does when the recognizer abstains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .rules import (
    Plan,
    UnplannableState,
    plan as make_plan,
    replan_after_failure,
    state_from_recognizer,
)


class Outcome(str, Enum):
    RUNNING = "running"
    DONE = "done"
    NEEDS_OPERATOR = "needs_operator"


# Subtasks whose result the recognizer can verify, and the probe that does it.
VERIFIED_BY = {"place": "tray", "flip": "mat"}
# What a passing verdict looks like for each.
EXPECTED = {"place": "ok", "flip": "flipped"}

# Flag a placement as bad above this probability. Chosen from the measured
# operating curve: recall 0.89 at precision 0.79, i.e. 9 false retries per
# 255 placements against 4 escaped defects. A false retry costs one re-pick;
# an escaped defect ships.
TRAY_BAD_THRESHOLD = 0.3
# Consecutive agreeing verdicts required before acting on one. Stops a
# single noisy frame from bouncing the plan.
HYSTERESIS = 2


@dataclass
class ManagerConfig:
    goal: str = "kit"
    max_retries_per_subtask: int = 3
    max_total_retries: int = 6
    tray_bad_threshold: float = TRAY_BAD_THRESHOLD
    hysteresis: int = HYSTERESIS
    # Value below which the policy is considered stalled rather than working.
    stall_value: float = -0.95
    # How long the value head may sit below `stall_value` before the manager
    # treats the attempt as failed without waiting for the boundary.
    stall_seconds: float = 6.0


@dataclass
class ManagerState:
    plan: Plan
    outcome: Outcome = Outcome.RUNNING
    retries: dict[str, int] = field(default_factory=dict)
    total_retries: int = 0
    reason: str = ""
    _pending: tuple[str, int] = ("", 0)
    _stalled_for: float = 0.0

    @property
    def subtask(self) -> str | None:
        return self.plan.subtasks[0] if self.plan.subtasks else None


class TaskManager:
    """Owns the plan and decides when to advance, retry, or stop."""

    def __init__(self, config: ManagerConfig | None = None):
        self.config = config or ManagerConfig()
        self.state: ManagerState | None = None

    # ───────────────────────────────────────────────────────────── start
    def begin(self, mat_label: str, holding: bool) -> ManagerState:
        object_state = state_from_recognizer(mat_label, holding)
        try:
            plan = make_plan(self.config.goal, object_state)
        except UnplannableState as exc:
            empty = Plan(goal=self.config.goal, state=object_state, task_prompt="", subtasks=[])
            self.state = ManagerState(
                plan=empty, outcome=Outcome.NEEDS_OPERATOR, reason=str(exc)
            )
            return self.state
        self.state = ManagerState(plan=plan)
        if plan.done:
            self.state.outcome = Outcome.DONE
        return self.state

    # ──────────────────────────────────────────────────── continuous tick
    def observe_value(self, value: float, dt: float) -> ManagerState:
        """Feed the 30 Hz value estimate; may declare a stall.

        Flip failures have no visible defect mid-attempt -- the gripper
        holds the part and nothing looks wrong -- so they present only as
        progress that never arrives. That is what this watches for.
        """
        state = self._require()
        if state.outcome is not Outcome.RUNNING:
            return state
        if value <= self.config.stall_value:
            state._stalled_for += dt
            if state._stalled_for >= self.config.stall_seconds and state.subtask:
                self._fail(state.subtask, "stalled: value flat at floor")
        else:
            state._stalled_for = 0.0
        return state

    # ───────────────────────────────────────────────── boundary checkpoint
    def subtask_finished(self, verdict: dict | None = None) -> ManagerState:
        """Called when the policy reports the current subtask complete.

        `verdict` is the recognizer output for subtasks it can check; None
        means unverifiable, in which case the manager takes the policy's
        word for it.
        """
        state = self._require()
        if state.outcome is not Outcome.RUNNING:
            return state
        subtask = state.subtask
        if subtask is None:
            state.outcome = Outcome.DONE
            return state

        probe = VERIFIED_BY.get(subtask)
        if probe is None or verdict is None:
            return self._advance()

        if verdict.get("label") == "abstain":
            state.outcome = Outcome.NEEDS_OPERATOR
            state.reason = f"recognizer abstained after {subtask} (ood={verdict.get('ood'):.1f})"
            return state

        passed = self._passes(subtask, verdict)
        decided = self._debounce(f"{subtask}:{passed}")
        if not decided:
            return state
        return self._advance() if passed else self._fail(subtask, "verification failed")

    # ───────────────────────────────────────────────────────── internals
    def _passes(self, subtask: str, verdict: dict) -> bool:
        probs = verdict.get("probs", {})
        if subtask == "place":
            return probs.get("bad", 0.0) <= self.config.tray_bad_threshold
        return verdict.get("label") == EXPECTED.get(subtask)

    def _debounce(self, key: str) -> bool:
        """True once the same verdict has repeated `hysteresis` times."""
        state = self._require()
        last, count = state._pending
        count = count + 1 if key == last else 1
        state._pending = (key, count)
        if count >= self.config.hysteresis:
            state._pending = ("", 0)
            return True
        return False

    def _advance(self) -> ManagerState:
        state = self._require()
        if state.plan.subtasks:
            state.plan.subtasks.pop(0)
        state._stalled_for = 0.0
        if state.plan.done:
            state.outcome = Outcome.DONE
        return state

    def _fail(self, subtask: str, reason: str) -> ManagerState:
        state = self._require()
        state.retries[subtask] = state.retries.get(subtask, 0) + 1
        state.total_retries += 1
        state._stalled_for = 0.0
        if (
            state.retries[subtask] > self.config.max_retries_per_subtask
            or state.total_retries > self.config.max_total_retries
        ):
            state.outcome = Outcome.NEEDS_OPERATOR
            state.reason = f"{subtask}: {reason}; retry budget exhausted"
            return state
        try:
            state.plan = replan_after_failure(state.plan, subtask)
        except UnplannableState as exc:
            state.outcome = Outcome.NEEDS_OPERATOR
            state.reason = str(exc)
            return state
        state.reason = f"{subtask}: {reason}; retrying via {state.plan.subtasks[0]}"
        return state

    def _require(self) -> ManagerState:
        if self.state is None:
            raise RuntimeError("call begin() before driving the manager")
        return self.state
