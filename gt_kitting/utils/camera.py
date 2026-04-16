#!/usr/bin/env python3
"""
Camera test script for kitting task.
Connects 4 cameras (2x RealSense D405, 2x Orbbec Gemini2) and displays via cv2.imshow.

Usage:
    LD_LIBRARY_PATH=/path/to/venv/lib/python3.12/site-packages:$LD_LIBRARY_PATH python gt_kitting/kitting.py
"""

import logging
import time
from dataclasses import dataclass
from threading import Event, Lock, Thread
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from pyorbbecsdk import (
    Config as OBConfig,
    Context as OBContext,
    OBFormat,
    OBSensorType,
    Pipeline,
    VideoStreamProfile,
)

from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.configs import ColorMode, Cv2Rotation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Orbbec Gemini2 camera class using pyorbbecsdk
# ---------------------------------------------------------------------------

@dataclass
class OrbbecCameraConfig:
    serial_number: str | None = None
    usb_port: str | None = None  # e.g. "6-1.3.2"
    fps: int = 30
    width: int = 1280
    height: int = 720
    warmup_s: float = 1.0

# ---------------------------------------------------------------------------
# Camera configurations matching training setup
# ---------------------------------------------------------------------------

CAMERA_CONFIGS = {
    "wrist_front": RealSenseCameraConfig(
        serial_number_or_name="315122272391",
        fps=30,
        width=848,
        height=480,
        color_mode=ColorMode.RGB,
        use_depth=False,
        rotation=Cv2Rotation.NO_ROTATION,
    ),
    "wrist_rear": RealSenseCameraConfig(
        serial_number_or_name="335122271613",
        fps=30,
        width=848,
        height=480,
        color_mode=ColorMode.RGB,
        use_depth=False,
        rotation=Cv2Rotation.ROTATE_180,
    ),
    "front_view": OrbbecCameraConfig(
        usb_port="6-1.3.3",  # SN: AY35C3200EM
        fps=30,
        width=1280,
        height=720,
    ),
    "side_view": OrbbecCameraConfig(
        usb_port="6-1.3.2",  # SN: AY3794301V0
        fps=30,
        width=1280,
        height=720,
    ),
}


class OrbbecCamera:
    """Orbbec Gemini2 color camera using pyorbbecsdk (color only, no depth)."""

    def __init__(self, config: OrbbecCameraConfig):
        self.config = config
        self.serial_number = config.serial_number
        self.usb_port = config.usb_port
        self.fps = config.fps
        self.width = config.width
        self.height = config.height
        self.warmup_s = config.warmup_s

        self._pipeline: Pipeline | None = None
        self._thread: Thread | None = None
        self._stop_event: Event | None = None
        self._frame_lock: Lock = Lock()
        self._latest_frame: NDArray[Any] | None = None
        self._latest_timestamp: float | None = None
        self._new_frame_event: Event = Event()

    def __str__(self) -> str:
        identifier = self.serial_number or self.usb_port
        return f"OrbbecCamera({identifier})"

    @property
    def is_connected(self) -> bool:
        return self._pipeline is not None

    def connect(self, max_retries: int = 3, retry_delay: float = 2.0) -> None:
        if self.is_connected:
            raise RuntimeError(f"{self} already connected.")

        # Find device by serial number or USB port with retries
        device = None
        for attempt in range(max_retries):
            try:
                ctx = OBContext()
                device_list = ctx.query_devices()
                for i in range(device_list.get_count()):
                    dev = device_list[i]
                    info = dev.get_device_info()
                    # Match by serial number or USB port (UID starts with port)
                    if self.serial_number and info.get_serial_number() == self.serial_number:
                        device = dev
                        break
                    if self.usb_port and info.get_uid().startswith(self.usb_port):
                        device = dev
                        self.serial_number = info.get_serial_number()
                        logger.info(f"Matched USB port {self.usb_port} -> SN {self.serial_number}")
                        break
                if device is not None:
                    break
            except Exception as e:
                logger.warning(f"{self} connect attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay}s...")
                time.sleep(retry_delay)

        if device is None:
            raise ConnectionError(
                f"Orbbec camera ({self}) not found after {max_retries} attempts."
            )

        self._pipeline = Pipeline(device)

        ob_config = OBConfig()
        profile_list = self._pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile: VideoStreamProfile = profile_list.get_video_stream_profile(
            self.width, self.height, OBFormat.RGB, self.fps
        )
        ob_config.enable_stream(color_profile)
        self._pipeline.start(ob_config)

        # Start background read thread
        self._stop_event = Event()
        self._thread = Thread(target=self._read_loop, name=f"{self}_read_loop", daemon=True)
        self._thread.start()

        # Warmup
        start = time.time()
        while time.time() - start < self.warmup_s:
            self.async_read(timeout_ms=int(self.warmup_s * 1000))
            time.sleep(0.1)

        with self._frame_lock:
            if self._latest_frame is None:
                raise ConnectionError(f"{self} failed to capture frames during warmup.")

        logger.info(f"{self} connected: {self.width}x{self.height} @ {self.fps}fps")

    def _read_loop(self) -> None:
        failure_count = 0
        while not self._stop_event.is_set():
            try:
                frames = self._pipeline.wait_for_frames(1000)
                if frames is None:
                    continue
                color_frame = frames.get_color_frame()
                if color_frame is None:
                    continue

                data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
                image = data.reshape((color_frame.get_height(), color_frame.get_width(), 3))

                with self._frame_lock:
                    self._latest_frame = image.copy()
                    self._latest_timestamp = time.perf_counter()
                self._new_frame_event.set()
                failure_count = 0

            except Exception as e:
                failure_count += 1
                if failure_count > 10:
                    logger.error(f"{self} exceeded max read failures: {e}")
                    break
                logger.warning(f"{self} read error: {e}")

    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        if not self.is_connected:
            raise RuntimeError(f"{self} not connected.")

        if not self._new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(f"Timed out waiting for frame from {self} after {timeout_ms}ms.")

        with self._frame_lock:
            frame = self._latest_frame
            self._new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"No frame available from {self}.")

        return frame

    def disconnect(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self._stop_event = None

        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None

        with self._frame_lock:
            self._latest_frame = None
            self._latest_timestamp = None
            self._new_frame_event.clear()

        logger.info(f"{self} disconnected.")

def create_cameras() -> dict[str, RealSenseCamera | OrbbecCamera]:
    """Create camera instances from configs."""
    cameras = {}
    for name, config in CAMERA_CONFIGS.items():
        if isinstance(config, RealSenseCameraConfig):
            cameras[name] = RealSenseCamera(config)
        elif isinstance(config, OrbbecCameraConfig):
            cameras[name] = OrbbecCamera(config)
    return cameras


def connect_cameras(cameras: dict[str, RealSenseCamera | OrbbecCamera]) -> None:
    """Connect all cameras. RealSense first, then Orbbec."""
    # Connect RealSense cameras first
    for name, camera in cameras.items():
        if isinstance(camera, RealSenseCamera):
            logger.info(f"Connecting {name}...")
            camera.connect()
    # Then Orbbec cameras
    for name, camera in cameras.items():
        if isinstance(camera, OrbbecCamera):
            logger.info(f"Connecting {name}...")
            camera.connect()


def disconnect_cameras(cameras: dict[str, RealSenseCamera | OrbbecCamera]) -> None:
    """Disconnect all cameras."""
    for name, camera in cameras.items():
        try:
            camera.disconnect()
        except Exception as e:
            logger.warning(f"Error disconnecting {name}: {e}")


def display_cameras(cameras: dict[str, RealSenseCamera | OrbbecCamera]) -> None:
    """Read frames from all cameras and display in a tiled window. Press 'q' to quit."""
    logger.info("Displaying camera feeds. Press 'q' to quit.")

    while True:
        frames = {}
        for name, camera in cameras.items():
            try:
                frame = camera.async_read(timeout_ms=1000)
                # RGB -> BGR for cv2.imshow
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frames[name] = frame_bgr
            except Exception as e:
                logger.warning(f"Failed to read {name}: {e}")

        if not frames:
            continue

        # Resize all frames to same height for tiling
        display_height = 360
        resized = {}
        for name, frame in frames.items():
            h, w = frame.shape[:2]
            scale = display_height / h
            new_w = int(w * scale)
            resized_frame = cv2.resize(frame, (new_w, display_height))
            cv2.putText(resized_frame, name, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            resized[name] = resized_frame

        # Tile: top row = wrist cameras, bottom row = external cameras
        order = ["wrist_front", "wrist_rear", "front_view", "side_view"]
        available = [resized[n] for n in order if n in resized]

        if len(available) >= 2:
            top_row = np.hstack(available[:2])
            if len(available) > 2:
                bottom_row = np.hstack(available[2:])
                # Pad to match widths
                w_diff = top_row.shape[1] - bottom_row.shape[1]
                if w_diff > 0:
                    pad = np.zeros((display_height, w_diff, 3), dtype=np.uint8)
                    bottom_row = np.hstack([bottom_row, pad])
                elif w_diff < 0:
                    pad = np.zeros((display_height, -w_diff, 3), dtype=np.uint8)
                    top_row = np.hstack([top_row, pad])
                tiled = np.vstack([top_row, bottom_row])
            else:
                tiled = top_row
        elif len(available) == 1:
            tiled = available[0]
        else:
            continue

        cv2.imshow("Kitting Cameras", tiled)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def main():
    cameras = create_cameras()
    try:
        connect_cameras(cameras)
        display_cameras(cameras)
    finally:
        disconnect_cameras(cameras)


if __name__ == '__main__':
    main()
