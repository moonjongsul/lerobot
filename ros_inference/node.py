"""ROS2 노드: 토픽 <-> RolloutController. 제어 로직은 여기 없다.

    python -m ros_inference.node --config ros_inference/config/xarm7_kitting.yaml

    ros2 service call /ros_inference/start std_srvs/srv/Trigger
    ros2 service call /ros_inference/stop  std_srvs/srv/Trigger
    ros2 topic pub --once /ros_inference/task std_msgs/String "data: pick"   # 다음 start의 과제

입력 (전부 BEST_EFFORT -- max_data_collect와 같은 이유: 액션·EEF 토픽이
BEST_EFFORT로 퍼블리시되고, RELIABLE 구독자는 거기에 조용히 연결되지 않는다)

    카메라 3대          sensor_msgs/CompressedImage
    /xarm/eef_pose     geometry_msgs/PoseStamped   (xarm_base, TCP)
    /xarm/eef_twist    geometry_msgs/TwistStamped  (xarm_base, TCP, m/s rad/s)
    그리퍼 상태         sensor_msgs/JointState

출력

    /xarm/vc_set_cartesian_velocity   xarm_msgs/srv/MoveVelocity, 30Hz, mm/s + rad/s,
                                      duration = command_duration_s (워치독)
    그리퍼 명령                        std_msgs/Float32, 값이 바뀔 때만
    /data_collection/rollout_action   max_interfaces/msg/TeleopAction, 30Hz (레코더용, m/s)
    /ros_inference/status             std_msgs/String (JSON), 1Hz + 상태 변화 시
    /ros_inference/task_complete      std_msgs/String (JSON), 메인 prompt 작업 종료 시 1회 (latched)

시작 시 mode 5(카르테시안 속도) + state 0을 설정하고, 끝나면 0 속도를 보낸 뒤
레코더를 SAVE한다(auto_record). 노드가 죽어도 duration 워치독으로 팔은 멈춘다.
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Float32, String
from std_srvs.srv import Trigger

from .action import Command, to_xarm_speeds
from .config import RolloutConfig, load_config
from .controller import RolloutController, ThreadedWorker
from .observation import RawObservation, decode_compressed
from .policy_runner import PolicyRunner

try:
    from xarm_msgs.srv import MoveVelocity, SetInt16
except ImportError as exc:  # pragma: no cover - 로봇 PC에는 있어야 한다
    raise ImportError("xarm_msgs가 필요하다 (xarm_ros2 워크스페이스를 source 할 것)") from exc

try:
    from max_interfaces.msg import TeleopAction
    from max_interfaces.srv import RecordCommand
except ImportError:  # 녹화 연동 없이도 제어는 돈다
    TeleopAction = RecordCommand = None

logger = logging.getLogger("ros_inference.node")


class _Latest:
    """스트림당 최신 메시지와 수신 시각(단조 시계)."""

    def __init__(self):
        self.msg = None
        self.recv = 0.0


class RolloutNode(Node):
    def __init__(self, cfg: RolloutConfig):
        super().__init__("ros_inference")
        self.cfg = cfg
        rc = cfg.ros
        self._lock = threading.Lock()
        self._latest: dict[str, _Latest] = {}
        self._warned: set[str] = set()

        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST,
                                durability=DurabilityPolicy.VOLATILE, depth=rc.qos_depth)
        subs = ReentrantCallbackGroup()
        for name, topic in rc.cameras.items():
            self._subscribe(f"cam/{name}", topic, CompressedImage, sensor_qos, subs)
        self._subscribe("pose", rc.eef_pose, PoseStamped, sensor_qos, subs)
        self._subscribe("twist", rc.eef_twist, TwistStamped, sensor_qos, subs)
        self._subscribe("gripper", rc.gripper_state, JointState, sensor_qos, subs)

        # 명령 경로
        cmd_group = MutuallyExclusiveCallbackGroup()
        self.vel_client = self.create_client(MoveVelocity, rc.velocity_service, callback_group=cmd_group)
        self.mode_client = self.create_client(SetInt16, rc.set_mode_service, callback_group=cmd_group)
        self.state_client = self.create_client(SetInt16, rc.set_state_service, callback_group=cmd_group)
        self.gripper_pub = self.create_publisher(Float32, rc.gripper_command, 10)
        self.action_pub = (
            self.create_publisher(TeleopAction, rc.rollout_action_topic, sensor_qos) if TeleopAction else None
        )
        self.record_client = (
            self.create_client(RecordCommand, rc.record_service) if RecordCommand and rc.auto_record else None
        )
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_pub = self.create_publisher(String, rc.status_topic, latched)
        self.complete_pub = self.create_publisher(String, rc.task_complete_topic, latched)
        self._inflight = 0

        # 정책
        self.runner = PolicyRunner(cfg.policy.checkpoint, cfg.policy.device, cfg.policy.num_inference_steps)
        missing = sorted(set(self.runner.camera_names) - set(rc.cameras))
        if missing:
            raise ValueError(f"policy needs cameras {missing} that the config does not subscribe")
        self.worker = ThreadedWorker(self.runner, cfg)
        self.recog_worker = None
        if self.runner.is_mvla and cfg.recognizer.path:
            from .recognition import Recognizer, RecognizerWorker

            rec = Recognizer(cfg.recognizer.path, cfg.recognizer.device, tuple(rc.image_size), rc.image_rotate)
            missing = sorted(set(rec.cameras) - set(rc.cameras))
            if missing:
                raise ValueError(f"recognizer needs cameras {missing} that the config does not subscribe")
            self.recog_worker = RecognizerWorker(rec)
        elif self.runner.is_mvla:
            self.get_logger().warn("no recognizer configured: start state from heads, no flip/place verification")
        self.ctrl = RolloutController(cfg, self.runner, self.worker, self._snapshot, recognizer=self.recog_worker)
        self.task = cfg.policy.task
        self._last_status = ""
        self._episode_log_path: Path | None = None

        # 제어
        ctl = MutuallyExclusiveCallbackGroup()
        self.create_service(Trigger, "~/start", self._on_start, callback_group=ctl)
        self.create_service(Trigger, "~/stop", self._on_stop, callback_group=ctl)
        self.create_subscription(String, "~/task", self._on_task, 10, callback_group=ctl)
        self.create_timer(1.0 / cfg.control.fps, self._on_tick, callback_group=ctl)
        self.create_timer(1.0, self._publish_status, callback_group=ctl)
        self.get_logger().info(
            f"ready: {self.runner.config.type} {self.runner.path} task={self.task!r} "
            f"speed_scale={cfg.safety.speed_scale}"
        )

    # ─────────────────────────────────────────────────────────── inputs
    def _subscribe(self, key, topic, msg_type, qos, group):
        self._latest[key] = _Latest()

        def cb(msg, _key=key):
            with self._lock:
                slot = self._latest[_key]
                slot.msg, slot.recv = msg, time.monotonic()

        self.create_subscription(msg_type, topic, cb, qos, callback_group=group)
        self.get_logger().info(f"subscribe {key:<18} {topic}")

    def _snapshot(self, with_images: bool) -> RawObservation | None:
        """최신 메시지들로 RawObservation. 하나라도 없으면 None.

        stamp는 가장 오래된 소스의 수신 시각 -- 컨트롤러의 정지 판단이 가장
        늦은 스트림을 기준으로 하도록.
        """
        with self._lock:
            latest = {k: (v.msg, v.recv) for k, v in self._latest.items()}
        if any(m is None for m, _ in latest.values()):
            return None
        pose, twist, grip = latest["pose"][0], latest["twist"][0], latest["gripper"][0]
        for label, msg in (("eef_pose", pose), ("eef_twist", twist)):
            fid = msg.header.frame_id
            if fid and fid != self.cfg.ros.expected_frame_id:
                self._warn_once(f"{label} frame_id {fid!r} != {self.cfg.ros.expected_frame_id!r}")
        stamp = min(r for _, r in latest.values())
        raw = RawObservation(
            stamp=stamp,
            eef_twist=np.array([twist.twist.linear.x, twist.twist.linear.y, twist.twist.linear.z,
                                twist.twist.angular.x, twist.twist.angular.y, twist.twist.angular.z]),
            eef_position=np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z]),
            eef_quat=np.array([pose.pose.orientation.x, pose.pose.orientation.y,
                               pose.pose.orientation.z, pose.pose.orientation.w]),
        )
        try:
            i = list(grip.name).index(self.cfg.ros.gripper_joint)
            raw.gripper_position = float(grip.position[i]) if i < len(grip.position) else 0.0
            raw.gripper_effort_raw = float(grip.effort[i]) if i < len(grip.effort) else 0.0
        except ValueError:
            self._warn_once(f"gripper joint {self.cfg.ros.gripper_joint!r} not in {list(grip.name)}")
            return None
        if with_images:
            for key, (msg, _) in latest.items():
                if key.startswith("cam/"):
                    img = decode_compressed(bytes(msg.data))
                    if img is None:
                        self._warn_once(f"{key}: undecodable frame")
                        return None
                    raw.images[key[4:]] = img
        return raw

    # ───────────────────────────────────────────────────────── control
    def _on_start(self, _req, resp):
        if self.ctrl.active:
            resp.success, resp.message = False, f"already {self.ctrl.status}"
            return resp
        if self._snapshot(False) is None:
            waiting = [k for k, v in self._latest.items() if v.msg is None]
            resp.success, resp.message = False, f"no data yet from {waiting}"
            return resp
        if self.cfg.ros.setup_velocity_mode and not self._setup_mode():
            resp.success, resp.message = False, "failed to set xArm mode 5 / state 0"
            return resp
        self._record(0)  # START
        self.ctrl.start(time.monotonic(), self.task)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        log_dir = Path(self.cfg.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self._episode_log_path = log_dir / f"rollout_{stamp}.jsonl"
        resp.success, resp.message = True, f"started task={self.task!r}"
        return resp

    def _on_stop(self, _req, resp):
        was = self.ctrl.active
        self.ctrl.stop("operator")
        self._send(self.ctrl.safety.stop())
        resp.success, resp.message = was, "stopped" if was else f"not running ({self.ctrl.status})"
        return resp

    def _on_task(self, msg: String):
        if self.ctrl.active:
            self.get_logger().warn("task change ignored while running")
            return
        self.task = msg.data.strip()
        self.get_logger().info(f"task set: {self.task!r}")

    def _on_tick(self):
        was_active = self.ctrl.active
        out = self.ctrl.tick(time.monotonic())
        if out.command is not None and (was_active or self.ctrl.active):
            self._send(out.command)
        if was_active and not self.ctrl.active:
            self._send(self.ctrl.safety.stop())
            if self.ctrl.completion is not None:
                # 메인 prompt의 작업 종료. 래치 토픽이라 늦게 붙은 구독자도 받는다.
                self.complete_pub.publish(String(data=json.dumps(self.ctrl.completion, default=float)))
                self.get_logger().info(f"task complete: {self.ctrl.completion}")
            self._record(2)  # SAVE -- 실패한 rollout도 다음 학습 데이터다
            self._write_log()
        if out.status != self._last_status or out.events:
            self._publish_status()

    def _setup_mode(self) -> bool:
        ok = True
        for client, value, label in ((self.mode_client, 5, "set_mode(5)"), (self.state_client, 0, "set_state(0)")):
            if not client.wait_for_service(timeout_sec=2.0):
                self.get_logger().error(f"{label}: service unavailable")
                return False
            req = SetInt16.Request()
            req.data = value
            future = client.call_async(req)
            deadline = time.monotonic() + 3.0
            while not future.done() and time.monotonic() < deadline:
                time.sleep(0.01)
            res = future.result() if future.done() else None
            if res is None or res.ret != 0:
                self.get_logger().error(f"{label} failed: {None if res is None else (res.ret, res.message)}")
                ok = False
        time.sleep(0.2)  # 모드 전환 직후의 명령은 무시될 수 있다
        return ok

    # ─────────────────────────────────────────────────────────── output
    def _send(self, cmd: Command) -> None:
        active = self.ctrl.active
        if self.vel_client.service_is_ready():
            if self._inflight > 5:
                self._warn_once("velocity service is falling behind (>5 calls in flight)")
            else:
                req = MoveVelocity.Request()
                req.speeds = to_xarm_speeds(cmd)
                req.is_tool_coord = False
                req.duration = float(self.cfg.control.command_duration_s)
                self._inflight += 1
                self.vel_client.call_async(req).add_done_callback(self._on_vel_done)
        else:
            self._warn_once(f"{self.cfg.ros.velocity_service} not available")

        if cmd.gripper is not None:
            self.gripper_pub.publish(Float32(data=float(cmd.gripper)))

        if self.action_pub is not None:
            msg = TeleopAction()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.space = TeleopAction.CARTESIAN_VELOCITY if active else TeleopAction.IDLE
            # 녹화 스키마와 같은 단위(m/s)로. 데모 녹화의 action 열과 바로 합쳐진다.
            msg.cartesian_velocity = [float(x) for x in cmd.twist]
            msg.is_tool_coord = False
            msg.gripper_width_percent = float(cmd.gripper_target)
            self.action_pub.publish(msg)

    def _on_vel_done(self, future):
        self._inflight = max(0, self._inflight - 1)
        res = future.result()
        if res is not None and res.ret != 0:
            self._warn_once(f"vc_set_cartesian_velocity ret={res.ret} {res.message}")

    def _record(self, command: int) -> None:
        if self.record_client is None:
            return
        if not self.record_client.service_is_ready():
            self._warn_once(f"recorder {self.cfg.ros.record_service} not available -- rollout not recorded")
            return
        req = RecordCommand.Request()
        req.command = command
        self.record_client.call_async(req)

    def _publish_status(self) -> None:
        snap = self.ctrl.snapshot(time.monotonic())
        snap.setdefault("task", self.task)
        self._last_status = self.ctrl.status
        self.status_pub.publish(String(data=json.dumps(snap, default=float)))

    def _write_log(self) -> None:
        if self._episode_log_path is None:
            return
        with self._episode_log_path.open("w") as f:
            f.write(json.dumps({"config": _asdict(self.cfg), "status": self.ctrl.status,
                                "reason": self.ctrl.reason, "completion": self.ctrl.completion},
                               default=float) + "\n")
            for row in self.ctrl.log:
                f.write(json.dumps(row, default=float) + "\n")
        self.get_logger().info(f"rollout log: {self._episode_log_path}")

    def _warn_once(self, message: str) -> None:
        if message not in self._warned:
            self._warned.add(message)
            self.get_logger().warn(message)

    def shutdown(self) -> None:
        self.ctrl.stop("shutdown")
        stop = self.ctrl.safety.stop()
        if self.vel_client.service_is_ready():
            req = MoveVelocity.Request()
            req.speeds = to_xarm_speeds(stop)
            req.is_tool_coord = False
            req.duration = 0.0
            self.vel_client.call_async(req)
        self.worker.close()
        if self.recog_worker is not None:
            self.recog_worker.close()


def _asdict(cfg):
    from dataclasses import asdict

    return asdict(cfg)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--task", default=None)
    p.add_argument("--speed-scale", type=float, default=None)
    args, ros_args = p.parse_known_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    overrides = {}
    if args.speed_scale is not None:
        overrides["safety"] = {"speed_scale": args.speed_scale}
    cfg = load_config(args.config, overrides)
    if args.checkpoint:
        cfg.policy.checkpoint = args.checkpoint
    if args.task:
        cfg.policy.task = args.task

    rclpy.init(args=ros_args)
    node = RolloutNode(cfg)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        time.sleep(0.2)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
