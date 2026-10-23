"""Hook-region crop for the cup-hang task, shared by dataset building and robot inference.

SAM3 finds the wooden pole of the mug tree (much more reliable than prompting the hooks themselves),
and a portrait crop around it is taken from the head and the wrist camera. The two crops are placed
side by side into one square image that is fed to the policy as an extra camera.

Per-episode state (call `reset()` at every episode start, both in training data generation and at inference):
- head: the camera and the mug tree are static, so the box from the first frame where the pole is
  detected is kept for the whole episode.
- wrist: the pole is detected on every call; when it is missed the previous box is kept.
"""

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

HOOK_IMAGE_KEY = "observation.images.hook"
PROMPT = "wooden pole"
OUT_SIZE = 224
PANEL_W, PANEL_H = OUT_SIZE // 2, OUT_SIZE


@dataclass
class CropParams:
    score_threshold: float = 0.5
    # crop height relative to the detected pole box height (pole box + margin above/below)
    height_scale: float = 1.15
    # minimum crop height as a fraction of the image height. The wrist view needs a larger value because
    # close to the mug tree the hooks stick out sideways well beyond a crop that only follows the pole.
    head_min_height_frac: float = 0.3
    wrist_min_height_frac: float = 0.7


def detect_poles(
    model: Sam3Model, processor: Sam3Processor, images: list[np.ndarray], score_threshold: float
) -> list[tuple[float, float, float, float] | None]:
    """Best `PROMPT` box per image (x1, y1, x2, y2) in pixels, or None when nothing passes the threshold."""
    device = model.device
    inputs = processor(images=images, text=[PROMPT] * len(images), return_tensors="pt").to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    with torch.no_grad():
        out = model(**inputs)
    results = processor.post_process_instance_segmentation(
        out, threshold=score_threshold, mask_threshold=0.5, target_sizes=inputs["original_sizes"].tolist()
    )
    boxes = []
    for r in results:
        if len(r["scores"]) == 0:
            boxes.append(None)
        else:
            best = int(r["scores"].argmax())
            boxes.append(tuple(float(v) for v in r["boxes"][best].tolist()))
    return boxes


def crop_rect(
    box: tuple[float, float, float, float] | None, img_h: int, img_w: int, min_height_frac: float, height_scale: float
) -> tuple[int, int, int, int]:
    """Portrait (1:2) crop rectangle (x1, y1, x2, y2) around the pole box, kept inside the image."""
    if box is None:
        h = img_h
        cx, cy = img_w / 2, img_h / 2
    else:
        x1, y1, x2, y2 = box
        h = max((y2 - y1) * height_scale, min_height_frac * img_h)
        h = min(h, img_h)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w = h * PANEL_W / PANEL_H
    left = min(max(cx - w / 2, 0), img_w - w)
    top = min(max(cy - h / 2, 0), img_h - h)
    return int(round(left)), int(round(top)), int(round(left + w)), int(round(top + h))


def compose(head: np.ndarray, wrist: np.ndarray, head_rect, wrist_rect) -> np.ndarray:
    """Side-by-side [head crop | wrist crop] as a uint8 (OUT_SIZE, OUT_SIZE, 3) image."""
    panels = []
    for img, (x1, y1, x2, y2) in ((head, head_rect), (wrist, wrist_rect)):
        crop = Image.fromarray(img[y1:y2, x1:x2]).resize((PANEL_W, PANEL_H), Image.BILINEAR)
        panels.append(np.asarray(crop))
    return np.concatenate(panels, axis=1)


class HookCropper:
    """Stateful per-episode cropper. Images are HxWx3 uint8 RGB arrays."""

    def __init__(self, device: str = "cuda", params: CropParams | None = None, model_id: str = "facebook/sam3"):
        self.params = params or CropParams()
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.model = Sam3Model.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device).eval()
        self.reset()

    def reset(self) -> None:
        self.head_box = None
        self.wrist_box = None

    def __call__(self, head: np.ndarray, wrist: np.ndarray) -> np.ndarray:
        if self.head_box is None:
            head_det, wrist_det = detect_poles(self.model, self.processor, [head, wrist], self.params.score_threshold)
            self.head_box = head_det
        else:
            (wrist_det,) = detect_poles(self.model, self.processor, [wrist], self.params.score_threshold)
        if wrist_det is not None:
            self.wrist_box = wrist_det
        return compose(head, wrist, *self.rects(head.shape[:2], wrist.shape[:2]))

    def add_to_observation(
        self,
        obs: dict[str, torch.Tensor],
        head_key: str = "observation.images.head",
        wrist_key: str = "observation.images.wrist",
    ) -> dict[str, torch.Tensor]:
        """Add HOOK_IMAGE_KEY to a policy observation whose images are float (1, 3, H, W) tensors in [0, 1]."""

        def to_np(t: torch.Tensor) -> np.ndarray:
            return (t[0].permute(1, 2, 0).clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()

        hook = self(to_np(obs[head_key]), to_np(obs[wrist_key]))
        obs[HOOK_IMAGE_KEY] = torch.from_numpy(hook).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return obs

    def rects(self, head_hw: tuple[int, int], wrist_hw: tuple[int, int]):
        p = self.params
        return (
            crop_rect(self.head_box, *head_hw, p.head_min_height_frac, p.height_scale),
            crop_rect(self.wrist_box, *wrist_hw, p.wrist_min_height_frac, p.height_scale),
        )
