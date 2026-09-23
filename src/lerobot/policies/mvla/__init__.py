"""MVLA: SmolVLA extended with state recognition, planning and value heads.

Three parts that are deliberately separable:

    recognizer/   a frozen encoder + linear probes answering "what state is
                  the scene in?". Independent of the policy on purpose --
                  a gate that shares the policy's representation shares its
                  blind spots, and this one guards against bad placements.

    planner/      recognised state -> task prompt + subtask sequence, and
                  the manager that drives it, verifies subtasks and
                  re-prompts on failure.

    (root)        the policy itself: SmolVLA plus subtask, value and status
                  heads, and the prompt assembly that conditions it.

Only the policy is registered with LeRobot; the other two are plain modules
so they can move out of the vendored tree without touching the policy.
"""

from .configuration_mvla import MVLAConfig
from .modeling_mvla import MVLAPolicy

__all__ = ["MVLAConfig", "MVLAPolicy"]
