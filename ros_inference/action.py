"""정책 출력 -> 로봇 명령. 안전 제한은 전부 여기 한 곳에 있다.

정책 출력 7채널은 데이터셋 action과 같은 의미다:

    [vx vy vz]   TCP 선속도, m/s, xarm_base 프레임 (베이스 좌표)
    [wx wy wz]   각속도, rad/s, 같은 프레임
    gripper      목표 폭 0(닫힘)..1(열림). 녹화에서는 0/1 두 값뿐이었다

단위 주의: `TeleopAction.msg` 주석은 선속도를 mm/s라고 하지만, 녹화된 값은
m/s다(데이터셋 action 범위 ±0.25). teleop 노드가 m/s로 퍼블리시하고 xArm
서비스를 부를 때 mm/s로 바꾸는 것으로 보인다. 그래서 정책 출력은 m/s로
다루고, `to_xarm_speeds`에서만 mm/s로 바꾼다.

`is_tool_coord`는 녹화되지 않았다. 베이스 프레임(False)으로 가정한다 --
운전자가 시연 중 툴 좌표로 바꿨다면 그 구간의 액션은 의미가 다르다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import SafetyConfig

MM_PER_M = 1000.0


@dataclass
class Command:
    linear: np.ndarray = field(default_factory=lambda: np.zeros(3))    # m/s
    angular: np.ndarray = field(default_factory=lambda: np.zeros(3))   # rad/s
    gripper: float | None = None   # 0.0 / 1.0, None = 이번 틱 변경 없음
    gripper_target: float = 1.0    # 현재 유지 중인 그리퍼 목표 (녹화용)
    flags: tuple[str, ...] = ()    # 어떤 제한이 걸렸는지

    @property
    def twist(self) -> np.ndarray:
        return np.concatenate([self.linear, self.angular])

    @property
    def is_zero(self) -> bool:
        return not np.any(self.linear) and not np.any(self.angular)


def to_xarm_speeds(cmd: Command) -> list[float]:
    """`vc_set_cartesian_velocity`의 speeds: [mm/s x3, rad/s x3]."""
    return [*(float(v) * MM_PER_M for v in cmd.linear), *(float(w) for w in cmd.angular)]


class ActionSafety:
    """정책 액션을 제한된 명령으로. 상태(직전 명령, 그리퍼)를 갖는다.

    적용 순서: 스케일 -> 축별 상한 -> 변화율 -> 작업공간. 작업공간을 맨
    뒤에 두는 건 그 제한이 감속 없이 즉시 걸려야 하기 때문이다.
    """

    def __init__(self, cfg: SafetyConfig, fps: float):
        self.cfg = cfg
        self.dt = 1.0 / fps
        self.ws_min = np.asarray(cfg.workspace_min, dtype=np.float64)
        self.ws_max = np.asarray(cfg.workspace_max, dtype=np.float64)
        self.reset()

    def reset(self, gripper_open: float = 1.0) -> None:
        self._last = np.zeros(6)
        self._gripper = gripper_open

    def stop(self) -> Command:
        """정지 명령. 직전 명령도 0으로 -- 재개 시 0에서 가속한다."""
        self._last = np.zeros(6)
        return Command(gripper_target=self._gripper, flags=("stop",))

    def __call__(self, action: np.ndarray, eef_position: np.ndarray) -> Command:
        cfg = self.cfg
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape[0] < 7 or not np.all(np.isfinite(a)):
            cmd = self.stop()
            cmd.flags = ("invalid_action",)
            return cmd
        flags: list[str] = []

        v = a[:6] * cfg.speed_scale
        lim = np.array([cfg.max_linear] * 3 + [cfg.max_angular] * 3)
        clipped = np.clip(v, -lim, lim)
        if np.any(clipped != v):
            flags.append("clip")
        v = clipped

        # 변화율 제한. 청크가 바뀌는 순간의 불연속이 로봇에 그대로 가지 않게.
        step = np.array([cfg.max_linear_accel] * 3 + [cfg.max_angular_accel] * 3) * self.dt
        limited = np.clip(v, self._last - step, self._last + step)
        if np.any(np.abs(limited - v) > 1e-9):
            flags.append("accel")
        v = limited

        # 작업공간: 경계 밖으로 향하는 선속도 성분만 0. 안쪽으로 돌아오는
        # 움직임은 막지 않는다 -- 막으면 경계에 붙은 채로 갇힌다.
        # 변화율 제한 뒤에 두어 즉시 걸리게 한다.
        p = np.asarray(eef_position, dtype=np.float64).reshape(3)
        out_low = (p <= self.ws_min) & (v[:3] < 0)
        out_high = (p >= self.ws_max) & (v[:3] > 0)
        if np.any(out_low | out_high):
            v[:3][out_low | out_high] = 0.0
            flags.append("workspace")

        self._last = v.copy()

        # 그리퍼: 히스테리시스 이진화. 바뀔 때만 명령을 낸다 -- teleop도
        # 버튼을 누를 때만 보냈고, 매 틱 재전송하면 그리퍼가 명령을 매번
        # 새로 시작할 수 있다.
        g = float(a[6])
        h = cfg.gripper_hysteresis
        new = self._gripper
        if g >= cfg.gripper_threshold + h:
            new = 1.0
        elif g <= cfg.gripper_threshold - h:
            new = 0.0
        changed = new != self._gripper
        self._gripper = new

        return Command(
            linear=v[:3], angular=v[3:],
            gripper=new if changed else None,
            gripper_target=new,
            flags=tuple(flags),
        )
