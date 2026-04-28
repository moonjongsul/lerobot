#!/usr/bin/env python3
"""
Kitting inference pipeline.
- Subscribes to /joint_states (7 joints) and /dxl_parallel_gripper/joint_states (gripper)
- Captures 4 camera frames (2x RealSense D405, 2x Orbbec Gemini2)
- Runs SmolVLA inference
- Publishes action to /gello/joint_states and /gripper/gripper_client/target_gripper_width_percent

Usage:
    LD_LIBRARY_PATH=.../site-packages:$LD_LIBRARY_PATH python gt_kitting/kitting.py
"""

import logging
import os
import signal
import sys
from threading import Lock

import numpy as np
import rclpy
import torch

os.environ["ROS_DOMAIN_ID"] = "0"
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

from utils.camera import (
    create_cameras,
    connect_cameras,
    disconnect_cameras,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- Config ----
CHECKPOINT_PATH = "/workspace/m.ax/checkpoints/smolvla_kitting_joint_b32/smolvla_kitting_scratch_b32/checkpoints/016000/pretrained_model"
# TASK_DESCRIPTION = "Pick part, flip or rotate to correct orientation, and place into kitting tray."
TASK_DESCRIPTION = "flip object"
INFERENCE_FPS = 30
DEVICE = "cuda"

INIT_POSITION = [
    0.8633871078491211,
    0.42438486218452454,
    0.18764179944992065,
    -1.3867534399032593,
    -0.05604249984025955,
    1.7663707733154297,
    1.8812178373336792,
]


class KittingInferenceNode(Node):
    def __init__(self, policy, preprocessor, postprocessor, cameras):
        super().__init__("kitting_inference")

        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.cameras = cameras

        # State buffer
        self._state_lock = Lock()
        self._joint_positions = None  # 7 joint values
        self._gripper_position = None  # 1 gripper value
        self._robot_ready = False  # True once robot topics are received

        # Subscribers
        self.create_subscription(
            JointState, "/joint_states", self._joint_states_cb, 10
        )
        self.create_subscription(
            JointState, "/dxl_parallel_gripper/joint_states", self._gripper_cb, 10
        )

        # Publishers
        self._joint_pub = self.create_publisher(JointState, "/gello/joint_states", 10)
        self._gripper_pub = self.create_publisher(
            Float32, "/gripper/gripper_client/target_gripper_width_percent", 10
        )

        # Inference timer
        period = 1.0 / INFERENCE_FPS
        self.create_timer(period, self._inference_loop)

        self.get_logger().info("Kitting inference node started.")

    # ---- Callbacks ----

    def _joint_states_cb(self, msg: JointState):
        with self._state_lock:
            self._joint_positions = list(msg.position[:7])

    def _gripper_cb(self, msg: JointState):
        with self._state_lock:
            if msg.position:
                self._gripper_position = msg.position[0]

    # ---- Inference ----

    def _get_state(self) -> np.ndarray | None:
        """Get current robot state as 8-dim array (7 joints + 1 gripper)."""
        with self._state_lock:
            if self._joint_positions is None or self._gripper_position is None:
                return None
            state = np.array(
                self._joint_positions + [self._gripper_position],
                dtype=np.float32,
            )
        return state

    def _get_images(self) -> dict[str, np.ndarray] | None:
        """Get current camera frames as RGB numpy arrays."""
        images = {}
        for name, camera in self.cameras.items():
            try:
                frame = camera.async_read(timeout_ms=500)
                images[name] = frame  # Already RGB
            except Exception as e:
                self.get_logger().warning(f"Camera {name} read failed: {e}")
                return None
        return images

    def _build_observation(self, state: np.ndarray, images: dict[str, np.ndarray]) -> dict:
        """Build observation dict for policy input."""
        obs = {
            "observation.state": torch.from_numpy(state).unsqueeze(0),
            "task": TASK_DESCRIPTION,
        }
        for cam_name, image in images.items():
            # image: (H, W, 3) uint8 RGB -> (1, 3, H, W) float [0,1]
            img_tensor = torch.from_numpy(image).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
            obs[f"observation.images.{cam_name}"] = img_tensor

        return obs

    def _publish_action(self, action: np.ndarray):
        """Publish action (8-dim: 7 joints + 1 gripper)."""
        # Joint command
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = [
            "fr3_joint1", "fr3_joint2", "fr3_joint3",
            "fr3_joint4", "fr3_joint5", "fr3_joint6", "fr3_joint7",
        ]
        joint_msg.position = action[:7].tolist()
        joint_msg.velocity = [0.0] * 7
        joint_msg.effort = [0.0] * 7
        self._joint_pub.publish(joint_msg)

        # Gripper command
        gripper_msg = Float32()
        gripper_msg.data = float(action[7])
        self._gripper_pub.publish(gripper_msg)

    def _publish_init_pose(self):
        """Publish initial pose to hold robot at start position."""
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = [
            "fr3_joint1", "fr3_joint2", "fr3_joint3",
            "fr3_joint4", "fr3_joint5", "fr3_joint6", "fr3_joint7",
        ]
        joint_msg.position = INIT_POSITION
        joint_msg.velocity = [0.0] * 7
        joint_msg.effort = [0.0] * 7
        self._joint_pub.publish(joint_msg)

    def _inference_loop(self):
        """Main inference loop called at INFERENCE_FPS."""
        state = self._get_state()
        if state is None:
            # Robot topics not yet received — publish init pose
            self._publish_init_pose()
            self.get_logger().info("Publishing init pose, waiting for robot state...", throttle_duration_sec=2.0)
            return

        if not self._robot_ready:
            self._robot_ready = True
            self.policy.reset()
            self.get_logger().info("Robot state received. Starting inference.")

        images = self._get_images()
        if images is None:
            return

        obs = self._build_observation(state, images)

        with torch.inference_mode():
            preprocessed = self.preprocessor(obs)
            action = self.policy.select_action(preprocessed)
            action_out = self.postprocessor(action)

        action_np = action_out.cpu().numpy().squeeze(0)  # (8,)
        self._publish_action(action_np)


def load_policy():
    """Load SmolVLA policy and processors from checkpoint."""
    logger.info(f"Loading policy from {CHECKPOINT_PATH}...")
    policy = SmolVLAPolicy.from_pretrained(CHECKPOINT_PATH)
    policy.to(DEVICE)
    policy.eval()
    logger.info("Policy loaded.")

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=CHECKPOINT_PATH,
    )
    logger.info("Preprocessor/postprocessor loaded.")

    return policy, preprocessor, postprocessor


def main():
    # 2. Connect cameras
    cameras = create_cameras()
    connect_cameras(cameras)
    # display_cameras(cameras)
    
    # 1. Load policy
    policy, preprocessor, postprocessor = load_policy()

    # 3. Start ROS2 node
    rclpy.init()
    node = KittingInferenceNode(policy, preprocessor, postprocessor, cameras)

    def shutdown(sig=None, frame=None):
        logger.info("Shutting down...")
        node.destroy_node()
        rclpy.try_shutdown()
        disconnect_cameras(cameras)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown()


if __name__ == "__main__":
    main()
