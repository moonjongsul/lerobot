"""상태 인식기 실행: 정책과 별도 모델, 별도 스레드.

설계 문서 §5.3의 세 번째 시계다. 정책(30Hz)과 value 헤드는 "잘 안 되고 있다"를
빠르게 말하고, 인식기는 경계에서 "부품이 제대로 놓였나 / 뒤집혔나"를 확정한다.
재시도나 완료를 결정하는 건 이 판정이다.

인식기는 학습 샘플과 같은 시점의 프레임을 봐야 한다(`recognizer/dataset.py`):

    mat   approach 런 시작 +3/12/21프레임 -- 팔이 아직 부품에서 떨어져 있을 때
    tray  그리퍼가 열린 순간 +8/18/28프레임 -- 손목이 아직 트레이 가까이 있을 때

그래서 tracker가 캡처 시각을 정하고, 컨트롤러가 그 시각의 관측을 떠서 여기로
넘긴다. 계산(DINOv2 + 선형 probe, 이미지 2장)은 별도 스레드에서 한다.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .observation import RawObservation, prepare_image

logger = logging.getLogger("ros_inference.recognition")


@dataclass
class Verdict:
    purpose: str          # start | flip | place
    t_obs: float
    result: dict          # {"label", "probs", "ood", "detected"}


class Recognizer:
    def __init__(self, path: str | Path, device: str = "cuda", image_size=(640, 480), rotate=None):
        from lerobot.policies.mvla.recognizer.model import StateRecognizer

        self.model = StateRecognizer.load(Path(path).expanduser())
        self.model.cfg.device = device
        self.image_size = tuple(image_size)
        self.rotate = rotate or {}
        _ = self.model.encoder  # 백본을 미리 올려 첫 판정이 늦지 않게
        logger.info(f"recognizer {path}: probes {sorted(self.model.probes)} cameras {self.model.cfg.cameras}")

    @property
    def cameras(self) -> tuple[str, ...]:
        return tuple(self.model.cfg.cameras)

    def __call__(self, task: str, raw: RawObservation) -> dict:
        # 녹화와 같은 640x480 BGR로 맞춘 뒤 넘긴다. 크롭은 모델이 한다.
        views = {c: prepare_image(raw.images[c], self.image_size, self.rotate.get(c, 0)) for c in self.cameras}
        out = self.model(task, views)
        if task == "mat" and not out["detected"]:
            # 부품 실루엣을 못 찾으면 크롭이 전체 프레임으로 떨어진다 -- 학습에
            # 없던 입력이므로 판정하지 않는다.
            out["label"] = "abstain"
        out["probs"] = {k: float(v) for k, v in out["probs"].items()}
        return out


PROBE_OF = {"start": "mat", "flip": "mat", "place": "tray"}


class RecognizerWorker:
    """실기용. 캡처된 관측을 받아 백그라운드에서 판정한다."""

    def __init__(self, recognizer: Recognizer):
        self.rec = recognizer
        self._in: queue.Queue = queue.Queue()
        self._out: queue.Queue[Verdict] = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name="recognizer", daemon=True)
        self._thread.start()

    def submit(self, purpose: str, raw: RawObservation, now: float) -> None:
        self._in.put((purpose, raw))

    def poll(self, now: float) -> list[Verdict]:
        out = []
        while True:
            try:
                out.append(self._out.get_nowait())
            except queue.Empty:
                return out

    def _loop(self) -> None:
        while True:
            item = self._in.get()
            if item is None:
                return
            purpose, raw = item
            try:
                self._out.put(Verdict(purpose, raw.stamp, self.rec(PROBE_OF[purpose], raw)))
            except Exception:  # noqa: BLE001
                logger.exception("recognizer failed")

    def close(self) -> None:
        self._in.put(None)


class SyncRecognizerWorker:
    """replay용. 즉시 계산하고 `latency_s` 뒤에 내준다."""

    def __init__(self, recognizer: Recognizer, latency_s: float = 0.05):
        self.rec = recognizer
        self.latency_s = latency_s
        self._pending: list[tuple[float, Verdict]] = []

    def submit(self, purpose: str, raw: RawObservation, now: float) -> None:
        self._pending.append((now + self.latency_s, Verdict(purpose, raw.stamp, self.rec(PROBE_OF[purpose], raw))))

    def poll(self, now: float) -> list[Verdict]:
        ready = [v for t, v in self._pending if t <= now]
        self._pending = [(t, v) for t, v in self._pending if t > now]
        return ready

    def close(self) -> None:
        pass


def aggregate(results: list[dict]) -> dict:
    """여러 프레임 판정을 하나로: 확률 평균, 과반이 기권이면 기권."""
    if not results:
        return {"label": "abstain", "probs": {}, "ood": float("nan")}
    abstain = sum(r["label"] == "abstain" for r in results)
    keys = list(results[0]["probs"])
    probs = {k: float(np.mean([r["probs"][k] for r in results])) for k in keys}
    label = "abstain" if abstain * 2 > len(results) else max(probs, key=probs.get)
    return {"label": label, "probs": probs, "ood": float(np.median([r["ood"] for r in results])),
            "frames": len(results)}
