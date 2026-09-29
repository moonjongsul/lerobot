"""체크포인트 로드와 추론. SmolVLA와 MVLA를 같은 인터페이스로.

전처리·후처리는 체크포인트에 저장된 파이프라인을 그대로 쓴다(카메라 이름
바꾸기, 토크나이즈, 정규화, 역정규화). 여기서 다시 구현하면 학습과 추론이
조용히 갈라진다.
"""

from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger("ros_inference.policy")

ELAPSED_KEY = "observation.elapsed_subtask"


@dataclass
class HeadRead:
    """MVLA 헤드 한 번 읽은 결과. 전부 파이썬 스칼라/배열."""

    subtask_probs: np.ndarray      # (7,) -- 6 라벨 + unlabelled
    value_subtask: float           # [-1, 0], 0 = 이 subtask 성공 직전
    value_subtask_spread: float
    value_episode: float
    status_probs: np.ndarray       # (3,) running / success / failure

    @property
    def subtask(self) -> int:
        return int(self.subtask_probs[:6].argmax())


def resolve_checkpoint(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if (p / "pretrained_model").is_dir():
        p = p / "pretrained_model"
    if not (p / "config.json").is_file():
        raise FileNotFoundError(f"no config.json under {p}")
    return p


class PolicyRunner:
    def __init__(self, checkpoint: str | Path, device: str = "cuda", num_inference_steps: int | None = None):
        import lerobot.policies  # noqa: F401 - config 서브클래스(mvla 포함) 등록
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        self.path = resolve_checkpoint(checkpoint)
        self.device = device
        cfg = PreTrainedConfig.from_pretrained(self.path)
        if num_inference_steps is not None:
            cfg.num_steps = int(num_inference_steps)
        cls = get_policy_class(cfg.type)
        t0 = time.time()
        self.policy = cls.from_pretrained(self.path, config=cfg)
        self.policy.to(device).eval()
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=str(self.path),
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.config = self.policy.config
        self.is_mvla = hasattr(self.policy, "head_outputs")
        self.camera_names = self._dataset_camera_names()
        logger.info(
            f"loaded {cfg.type} from {self.path} in {time.time() - t0:.1f}s "
            f"(chunk {self.config.chunk_size}, steps {self.config.num_steps}, cameras {self.camera_names})"
        )

    def _dataset_camera_names(self) -> list[str]:
        """정책이 기대하는 카메라를 데이터셋 이름으로.

        정책 feature는 camera1/2/3이고, 저장된 rename 단계가 데이터셋
        이름 -> cameraN으로 바꾼다. 그 표를 거꾸로 읽는다.
        """
        rename = {}
        for step in getattr(self.pre, "steps", []):
            m = getattr(step, "rename_map", None)
            if m:
                rename.update(m)
        inverse = {v: k for k, v in rename.items()}
        names = []
        for key in self.config.image_features:
            src = inverse.get(key, key)
            names.append(src.removeprefix("observation.images."))
        return names

    def _autocast(self):
        if self.device.startswith("cuda"):
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _batch(self, obs: dict[str, torch.Tensor], task: str, elapsed: float | None) -> dict:
        frame = dict(obs)
        frame["task"] = task
        if self.is_mvla and self.config.use_elapsed_subtask:
            frame[ELAPSED_KEY] = torch.tensor([float(elapsed or 0.0) / self.config.elapsed_scale])
        return self.pre(frame)

    @torch.no_grad()
    def infer_chunk(self, obs: dict[str, torch.Tensor], task: str, elapsed: float | None = None,
                    seed: int | None = None) -> np.ndarray:
        """(chunk_size, 7) 역정규화된 액션. m/s, rad/s, gripper 0..1."""
        batch = self._batch(obs, task, elapsed)
        noise = None
        if seed is not None:
            g = torch.Generator(device=self.device).manual_seed(seed)
            noise = torch.randn((1, self.config.chunk_size, self.config.max_action_dim),
                                generator=g, device=self.device)
        self.policy.reset()  # 큐 비우기 -- 청크 관리는 호출자(ChunkExecutor)가 한다
        with self._autocast():
            chunk = self.policy.predict_action_chunk(batch, noise=noise)
        chunk = self.post(chunk.float())
        return chunk.squeeze(0).cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def read_heads(self, obs: dict[str, torch.Tensor], task: str, elapsed: float | None) -> HeadRead | None:
        if not self.is_mvla:
            return None
        batch = self._batch(obs, task, elapsed)
        with self._autocast():
            out = self.policy.head_outputs(batch)
        heads = self.policy.heads
        return HeadRead(
            subtask_probs=out["subtask_logits"].float().softmax(-1)[0].cpu().numpy(),
            value_subtask=float(heads.value_subtask.expected_value(out["value_subtask_logits"])[0]),
            value_subtask_spread=float(heads.value_subtask.spread(out["value_subtask_logits"])[0]),
            value_episode=float(heads.value_episode.expected_value(out["value_episode_logits"])[0]),
            status_probs=out["status_logits"].float().softmax(-1)[0].cpu().numpy(),
        )
