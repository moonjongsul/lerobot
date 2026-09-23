"""The state recognizer: a frozen encoder plus a linear probe per task.

Deliberately small. Seven backbones were compared on this data -- DINOv2
base/large/registers, SigLIP2, SigLIP-so400m, CLIP-L, AIMv2 -- and they land
within one standard deviation of each other (AUC 0.986-0.989 on mat_state).
Input resolution 224 -> 448 moves AUC by 0.001. Swapping the linear probe
for an MLP or kNN makes things worse: with 234 episodes they overfit. The
ceiling here is the data, not the representation, so the smallest encoder
that reaches it is the right one.

Sharing SmolVLA's own vision tower was measured and rejected: that tower is
86.4M parameters, the same size as DINOv2-base, so sharing saves nothing
measurable at the 0.3 Hz this runs at, while scoring lower on mat_state
(95.7% vs 97.9%) and doubling the background leak. Reading its KV cache is
worse still -- after pixel shuffle the object spans about 2x2 tokens.

Staying independent also keeps the failure modes of the place gate
uncorrelated with the policy it is gating.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

DEFAULT_BACKBONE = "facebook/dinov2-base"


@dataclass
class RecognizerConfig:
    backbone: str = DEFAULT_BACKBONE
    image_size: int = 224
    cameras: tuple[str, ...] = ("wrist_front", "wrist_rear")
    # Probe strength. C=0.1 was the best compromise across both tasks; the
    # metric is flat between 0.01 and 1.
    probe_c: float = 0.1
    # Flag a prediction as "do not know" when the pooled feature sits this
    # many robust standard deviations from the training distribution.
    ood_threshold: float = 6.0
    device: str = "cuda"
    dtype: str = "float16"
    batch_size: int = 32
    tasks: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "mat": ("empty", "target", "flipped", "unknown_pose"),
            "tray": ("ok", "bad"),
        }
    )


class FrozenEncoder:
    """Mean-pooled patch tokens from a frozen vision backbone."""

    def __init__(self, cfg: RecognizerConfig):
        from transformers import AutoImageProcessor, AutoModel

        self.cfg = cfg
        self.processor = AutoImageProcessor.from_pretrained(cfg.backbone)
        model = AutoModel.from_pretrained(cfg.backbone, dtype=getattr(torch, cfg.dtype))
        # CLIP/SigLIP wrap their tower; DINOv2 is one already.
        self.model = getattr(model, "vision_model", model).to(cfg.device).eval()

    @torch.no_grad()
    def __call__(self, images: list[np.ndarray]) -> np.ndarray:
        """RGB uint8 arrays -> (N, D) float32 features."""
        out = []
        bs = self.cfg.batch_size
        for i in range(0, len(images), bs):
            pil = [Image.fromarray(im) for im in images[i : i + bs]]
            batch = self.processor(images=pil, return_tensors="pt").to(self.cfg.device)
            pixel = batch["pixel_values"].to(getattr(torch, self.cfg.dtype))
            hidden = self.model(pixel_values=pixel).last_hidden_state.float()
            out.append(hidden.mean(1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 0), np.float32)


class LinearProbe:
    """Logistic regression over frozen features, with an abstain rule.

    The abstain rule exists because `unknown_pose` -- an object lying in
    neither canonical pose -- has no training examples, and the genuinely
    unexpected (a dropped part, a hand in frame) cannot be enumerated at
    all. A closed-set classifier answers confidently anyway. Distance from
    the training feature distribution is a claim the model can actually
    support: "this does not look like anything I was fitted on".
    """

    def __init__(self, cfg: RecognizerConfig, classes: tuple[str, ...]):
        self.cfg = cfg
        self.classes = list(classes)
        self.pipeline = None
        self.seen: list[str] = []
        self._center: np.ndarray | None = None
        self._scale: np.ndarray | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> LinearProbe:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.seen = sorted(set(labels.tolist()))
        self.pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=4000, C=self.cfg.probe_c, class_weight="balanced"
            ),
        )
        self.pipeline.fit(features, labels)
        # Median / MAD rather than mean / std: robust to the tail of odd
        # frames that always exist in real recordings.
        self._center = np.median(features, axis=0)
        mad = np.median(np.abs(features - self._center), axis=0)
        self._scale = np.maximum(mad * 1.4826, 1e-6)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """(N, len(classes)) probabilities, zero-filled for unseen classes."""
        raw = self.pipeline.predict_proba(features)
        out = np.zeros((len(features), len(self.classes)), dtype=np.float64)
        for j, name in enumerate(self.pipeline.classes_):
            out[:, self.classes.index(name)] = raw[:, j]
        return out

    def ood_score(self, features: np.ndarray) -> np.ndarray:
        """Robust z-distance from the training features; higher is stranger."""
        z = np.abs(features - self._center) / self._scale
        return np.median(z, axis=1)

    def predict(self, features: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray]:
        """(labels, probabilities, ood). Label is "abstain" past the threshold."""
        probs = self.predict_proba(features)
        ood = self.ood_score(features)
        names = [
            "abstain" if o > self.cfg.ood_threshold else self.classes[int(p.argmax())]
            for p, o in zip(probs, ood, strict=True)
        ]
        return names, probs, ood

    def save(self, path: Path) -> None:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "classes": self.classes,
                "seen": self.seen,
                "center": self._center,
                "scale": self._scale,
                "cfg": self.cfg.__dict__,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> LinearProbe:
        import joblib

        blob = joblib.load(path)
        cfg = RecognizerConfig(**blob["cfg"])
        probe = cls(cfg, tuple(blob["classes"]))
        probe.pipeline = blob["pipeline"]
        probe.seen = blob["seen"]
        probe._center = blob["center"]
        probe._scale = blob["scale"]
        return probe


class StateRecognizer:
    """Both probes behind one call, for use by the task manager."""

    def __init__(self, cfg: RecognizerConfig, probes: dict[str, LinearProbe]):
        self.cfg = cfg
        self.probes = probes
        self._encoder: FrozenEncoder | None = None

    @property
    def encoder(self) -> FrozenEncoder:
        if self._encoder is None:
            self._encoder = FrozenEncoder(self.cfg)
        return self._encoder

    def encode_views(self, views: dict[str, np.ndarray]) -> np.ndarray:
        """Concatenate per-camera features in a fixed camera order."""
        ordered = [views[c] for c in self.cfg.cameras]
        feats = self.encoder(ordered)
        return feats.reshape(1, -1)

    def __call__(self, task: str, views: dict[str, np.ndarray]) -> dict:
        features = self.encode_views(views)
        names, probs, ood = self.probes[task].predict(features)
        return {
            "label": names[0],
            "probs": dict(zip(self.probes[task].classes, probs[0], strict=True)),
            "ood": float(ood[0]),
        }

    @classmethod
    def load(cls, directory: Path) -> StateRecognizer:
        directory = Path(directory)
        cfg = RecognizerConfig(**json.loads((directory / "config.json").read_text()))
        probes = {
            p.stem.replace("probe_", ""): LinearProbe.load(p)
            for p in sorted(directory.glob("probe_*.joblib"))
        }
        return cls(cfg, probes)

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text(json.dumps(self.cfg.__dict__, indent=2))
        for task, probe in self.probes.items():
            probe.save(directory / f"probe_{task}.joblib")
