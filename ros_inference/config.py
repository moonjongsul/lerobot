"""YAML -> 타입 있는 설정. ROS 파라미터가 아니라 파일 하나로 둔다.

ROS 파라미터로 두면 노드 밖(replay, 테스트)에서 같은 설정을 읽을 수 없다.
rollout 결과를 재현하려면 실기와 오프라인이 한 파일을 봐야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class PolicyConfig:
    checkpoint: str = ""
    device: str = "cuda"
    task: str = "kit"
    num_inference_steps: int | None = None


@dataclass
class ControlConfig:
    fps: float = 30.0
    execute_steps: int = 25
    blend_steps: int = 5
    max_episode_s: float = 180.0
    stale_timeout_s: float = 0.3
    command_duration_s: float = 0.2


@dataclass
class SafetyConfig:
    speed_scale: float = 0.5
    max_linear: float = 0.25
    max_angular: float = 0.6
    max_linear_accel: float = 1.5
    max_angular_accel: float = 4.0
    workspace_min: list[float] = field(default_factory=lambda: [-0.32, -0.61, 0.0])
    workspace_max: list[float] = field(default_factory=lambda: [0.65, 0.12, 0.35])
    gripper_threshold: float = 0.5
    gripper_hysteresis: float = 0.15


@dataclass
class MVLARuntimeConfig:
    initial_state: str = "auto"
    holding_effort: float = 0.3
    hysteresis: int = 2
    head_hz: float = 5.0
    head_prompt: str = "self"
    speed_bin: int | None = None
    # 메인 prompt 작업 종료: plan의 마지막 subtask에서 아래 중 하나가 연속 N회
    completion_value_subtask: float = -0.05
    completion_value_episode: float = -0.05
    completion_consecutive: int = 3


@dataclass
class RecognizerRuntimeConfig:
    # 비우면 인식기 없이 돈다 (시작 상태는 헤드로, 검증 없이 완료)
    path: str = ""
    device: str = "cuda"
    verify_flip: bool = True
    verify_place: bool = True
    # 캡처 시각. 인식기 학습 샘플과 같은 시점이어야 한다 (recognizer/dataset.py)
    start_offsets_s: list[float] = field(default_factory=lambda: [0.1, 0.4, 0.7])
    flip_offsets_frames: list[int] = field(default_factory=lambda: [3, 12, 21])
    release_offsets_frames: list[int] = field(default_factory=lambda: [8, 18, 28])
    # place 중 관측 그리퍼 위치가 최저값에서 이만큼 오르면 릴리스
    release_rise: float = 0.15
    # 정책이 place를 끝냈다고 보는데 릴리스가 안 보이면 이만큼 기다린 뒤 검증 없이 완료
    unverified_timeout_s: float = 5.0
    # tray P(bad)가 이보다 크면 불량. 0.3 = 설계 권장(놓친 불량 최소화).
    # held-out val(bad 7건): 0.3 -> 6/7 탐지, 오경보 2 / 0.5 -> 6/7, 오경보 0
    tray_bad_threshold: float = 0.3


@dataclass
class RosConfig:
    qos_depth: int = 10
    cameras: dict[str, str] = field(default_factory=dict)
    image_size: list[int] = field(default_factory=lambda: [640, 480])
    image_rotate: dict[str, int] = field(default_factory=dict)
    eef_pose: str = "/xarm/eef_pose"
    eef_twist: str = "/xarm/eef_twist"
    expected_frame_id: str = "xarm_base"
    gripper_state: str = "/dxl_parallel_gripper/joint_states"
    gripper_joint: str = "finger_joint"
    gripper_effort_max: float = 600.0
    velocity_service: str = "/xarm/vc_set_cartesian_velocity"
    set_mode_service: str = "/xarm/set_mode"
    set_state_service: str = "/xarm/set_state"
    setup_velocity_mode: bool = True
    gripper_command: str = "/gripper/gripper_client/target_gripper_width_percent"
    rollout_action_topic: str = "/data_collection/rollout_action"
    record_service: str = "/data_collection/record_command"
    auto_record: bool = True
    status_topic: str = "/ros_inference/status"
    task_complete_topic: str = "/ros_inference/task_complete"


@dataclass
class RolloutConfig:
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    mvla: MVLARuntimeConfig = field(default_factory=MVLARuntimeConfig)
    recognizer: RecognizerRuntimeConfig = field(default_factory=RecognizerRuntimeConfig)
    ros: RosConfig = field(default_factory=RosConfig)
    log_dir: str = "rollout_logs"


def _build(cls, data: dict | None):
    data = dict(data or {})
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        # 오타 난 키가 조용히 기본값으로 대체되면, 안전 설정이 적용된 줄
        # 알고 로봇을 돌리게 된다.
        raise KeyError(f"{cls.__name__}: unknown key(s) {unknown}")
    return cls(**data)


def load_config(path: str | Path, overrides: dict | None = None) -> RolloutConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    for section, values in (overrides or {}).items():
        raw.setdefault(section, {}).update(values)
    top = {k: v for k, v in raw.items() if k not in ("policy", "control", "safety", "mvla", "recognizer", "ros")}
    return RolloutConfig(
        policy=_build(PolicyConfig, raw.get("policy")),
        control=_build(ControlConfig, raw.get("control")),
        safety=_build(SafetyConfig, raw.get("safety")),
        mvla=_build(MVLARuntimeConfig, raw.get("mvla")),
        recognizer=_build(RecognizerRuntimeConfig, raw.get("recognizer")),
        ros=_build(RosConfig, raw.get("ros")),
        **top,
    )
