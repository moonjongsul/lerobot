"""Held-out 평가: 보조 헤드가 실제로 무엇을 알아냈는지 잰다.

학습 손실은 이 질문에 답하지 못한다. status 헤드를 보면 분명하다 --
터미널 프레임은 런당 하나뿐이라 99.52%가 `running`이고, "항상 running"이라
답하기만 해도 정확도 99.5%가 나온다. 정확도를 보고하면 붕괴한 헤드와
동작하는 헤드가 똑같이 훌륭해 보인다.

그래서 여기서 재는 것은 정확도가 아니라 **판별력**이다:

    subtask      7-class 정확도 + macro-F1 (최빈 23.3%로 균형적이라 정확도가
                 의미를 갖는 유일한 헤드)
    value        실패 런 탐지 AUC. 헤드의 존재 이유 그 자체다 -- 실패한
                 시도의 값이 성공한 시도보다 낮게 나오는가
    리드타임     실패 런에서 값이 임계 아래로 내려간 시점과 런 종료 사이의
                 시간. 양수여야 조기 경보로 쓸 수 있다
    status       실패 클래스 AUC. 정확도는 위 이유로 보고하지 않는다

모든 집계는 **런 단위**다. 프레임 단위로 세면 긴 런이 짧은 런보다 결과를
많이 좌우하고, 실패 런 94건이 283,194개의 running 프레임에 묻힌다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .modeling_mvla import ELAPSED_KEY, TARGET_KEYS, TARGET_NAMES

# 실패 경보를 울리는 값 임계. 리드타임은 이 값에 의존하므로 여러 개를 본다.
LEAD_TIME_THRESHOLDS = (-0.8, -0.6, -0.4)


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC. 순위 통계라 클래스 불균형에 영향받지 않는다.

    sklearn을 부르지 않는다. 이 한 줄 때문에 학습 환경에 의존성을 늘릴
    이유가 없고, Mann-Whitney U 형태가 동점 처리까지 그대로 맞다.
    """
    pos, neg = labels == 1, labels == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = scores.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # 동점은 평균 순위로 -- 없으면 붕괴한 헤드(전부 같은 값)의 AUC가
    # 0.5가 아니라 입력 순서에 따라 아무 값이나 나온다.
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _macro_f1(pred: np.ndarray, true: np.ndarray, n_classes: int) -> float:
    """클래스별 F1의 단순 평균. 드문 클래스를 무시하면 점수가 떨어진다."""
    f1s = []
    for c in range(n_classes):
        tp = int(((pred == c) & (true == c)).sum())
        fp = int(((pred == c) & (true != c)).sum())
        fn = int(((pred != c) & (true == c)).sum())
        if tp + fn == 0:  # 정답에 없는 클래스는 평균에서 뺀다
            continue
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(f1s)) if f1s else float("nan")


@dataclass
class HeadMetrics:
    """한 번의 held-out 통과 결과. 전부 스칼라라 그대로 wandb에 들어간다."""

    n_frames: int = 0
    n_runs: int = 0
    n_failed_runs: int = 0
    scalars: dict[str, float] = field(default_factory=dict)

    def to_wandb(self, prefix: str = "val") -> dict[str, float]:
        out = {f"{prefix}/{k}": v for k, v in self.scalars.items() if not np.isnan(v)}
        out[f"{prefix}/n_frames"] = self.n_frames
        out[f"{prefix}/n_runs"] = self.n_runs
        return out

    def summary(self) -> str:
        parts = [f"val {self.n_frames}frames/{self.n_runs}runs({self.n_failed_runs}failed)"]
        for k in sorted(self.scalars):
            v = self.scalars[k]
            if not np.isnan(v):
                parts.append(f"{k}:{v:.3f}")
        return "  ".join(parts)


@torch.no_grad()
def collect_predictions(
    policy,
    dataset,
    preprocessor,
    *,
    batch_size: int = 32,
    num_workers: int = 4,
    max_batches: int | None = None,
    device=None,
) -> dict[str, np.ndarray]:
    """held-out 데이터셋 전체에 대해 헤드 출력과 정답을 모은다.

    `policy.forward`가 아니라 `head_outputs`만 부른다 -- flow matching은
    여기서 필요 없고, 그 denoising 비용이 평가 시간의 대부분을 차지한다.
    """
    was_training = policy.training
    policy.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device is not None and device.type == "cuda",
        drop_last=False,
    )

    buf: dict[str, list[np.ndarray]] = {}

    def push(key: str, value: Tensor) -> None:
        buf.setdefault(key, []).append(value.detach().float().cpu().numpy())

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = preprocessor(batch)
        outputs = policy.head_outputs(batch)

        if "subtask_logits" in outputs:
            push("subtask_pred", outputs["subtask_logits"].argmax(-1))
        if "value_subtask_logits" in outputs:
            push("value_subtask_pred", policy.heads.value_subtask.expected_value(
                outputs["value_subtask_logits"]))
            push("value_subtask_spread", policy.heads.value_subtask.spread(
                outputs["value_subtask_logits"]))
        if "value_episode_logits" in outputs:
            push("value_episode_pred", policy.heads.value_episode.expected_value(
                outputs["value_episode_logits"]))
        if "status_logits" in outputs:
            probs = outputs["status_logits"].softmax(-1)
            push("status_pred", outputs["status_logits"].argmax(-1))
            push("status_p_failure", probs[:, 2])

        for key, name in zip(TARGET_KEYS, TARGET_NAMES, strict=True):
            if key in batch:
                push(f"target_{name}", batch[key])
        if ELAPSED_KEY in batch:
            push("elapsed", batch[ELAPSED_KEY])
        # 런 단위 집계를 위한 식별자. 전처리가 통과시키는 키들이다.
        for key in ("index", "episode_index"):
            if key in batch:
                push(key, batch[key])

    if was_training:
        policy.train()

    return {k: np.concatenate(v) for k, v in buf.items() if v}


def compute_metrics(
    preds: dict[str, np.ndarray],
    runs_by_index: dict[int, tuple[int, bool]] | None = None,
    fps: float = 30.0,
    elapsed_scale: float = 30.0,
) -> HeadMetrics:
    """모은 예측을 지표로 바꾼다.

    `runs_by_index`는 전역 프레임 인덱스 -> (run_id, failed). 없으면 런
    단위 지표(실패 탐지 AUC, 리드타임)는 건너뛰고 프레임 단위만 낸다.
    """
    m = HeadMetrics()
    m.n_frames = len(next(iter(preds.values()))) if preds else 0
    if not m.n_frames:
        return m

    # ── 프레임 단위 ────────────────────────────────────────────────
    if "subtask_pred" in preds and "target_subtask_index" in preds:
        pred = preds["subtask_pred"].astype(int)
        true = preds["target_subtask_index"].astype(int)
        valid = true >= 0
        if valid.any():
            m.scalars["subtask_acc"] = float((pred[valid] == true[valid]).mean())
            m.scalars["subtask_macro_f1"] = _macro_f1(pred[valid], true[valid], n_classes=7)

    for name in ("value_subtask", "value_episode"):
        if f"{name}_pred" in preds and f"target_{name}" in preds:
            err = preds[f"{name}_pred"] - preds[f"target_{name}"]
            m.scalars[f"{name}_mae"] = float(np.abs(err).mean())

    # ── 런 단위 ───────────────────────────────────────────────────
    if runs_by_index is None or "index" not in preds:
        return m

    idx = preds["index"].astype(int)
    known = np.array([i in runs_by_index for i in idx])
    if not known.any():
        return m
    run_id = np.array([runs_by_index[i][0] if k else -1 for i, k in zip(idx, known)])
    failed = np.array([runs_by_index[i][1] if k else False for i, k in zip(idx, known)])

    uniq = np.unique(run_id[known])
    m.n_runs = len(uniq)
    m.n_failed_runs = int(sum(failed[run_id == r][0] for r in uniq))

    # 실패 탐지: 런당 최저 예측값을 점수로. 실패는 어느 순간 값이 떨어지면
    # 되는 것이지 내내 낮아야 하는 게 아니다.
    if "value_subtask_pred" in preds:
        scores, labels = [], []
        for r in uniq:
            sel = run_id == r
            scores.append(-float(preds["value_subtask_pred"][sel].min()))  # 낮을수록 실패 -> 부호 반전
            labels.append(int(failed[sel][0]))
        m.scalars["failure_auc_value"] = _auc(np.array(scores), np.array(labels))

    if "status_p_failure" in preds:
        scores, labels = [], []
        for r in uniq:
            sel = run_id == r
            scores.append(float(preds["status_p_failure"][sel].max()))
            labels.append(int(failed[sel][0]))
        m.scalars["failure_auc_status"] = _auc(np.array(scores), np.array(labels))

    # 리드타임: 실패 런에서 경보가 울린 뒤 런이 끝나기까지 남은 시간.
    # 음수면 런이 끝난 뒤에야 알아차렸다는 뜻이고, 그건 경보가 아니다.
    if "value_subtask_pred" in preds and "elapsed" in preds:
        for thr in LEAD_TIME_THRESHOLDS:
            leads = []
            for r in uniq:
                sel = run_id == r
                if not failed[sel][0]:
                    continue
                v = preds["value_subtask_pred"][sel]
                t = preds["elapsed"][sel] * elapsed_scale  # 초로 되돌린다
                below = np.flatnonzero(v < thr)
                if len(below) == 0:
                    leads.append(0.0)  # 끝까지 경보 없음
                    continue
                leads.append(float(t.max() - t[below[0]]))
            if leads:
                m.scalars[f"lead_s@{abs(thr):.1f}"] = float(np.mean(leads))
                m.scalars[f"lead_detected@{abs(thr):.1f}"] = float(np.mean([l > 0 for l in leads]))

    return m


def build_run_lookup(root) -> dict[int, tuple[int, bool]]:
    """전역 프레임 인덱스 -> (run_id, 이 런이 실패했는가).

    `derive_frame_targets`가 이미 프레임을 런에 배정해 두었으므로 그것을
    쓴다. 여기서 경계를 다시 계산하면 정의가 두 벌이 된다.
    """
    from .data.derive import derive_frame_targets
    from .data.segments import runs_frame

    targets = derive_frame_targets(root)
    failed_by_run = dict(zip(runs_frame(root)["run_id"], runs_frame(root)["failed"], strict=True))
    # derive_frame_targets는 (episode, frame) 순으로 정렬되어 있고, 데이터셋의
    # 전역 `index` 컬럼도 같은 순서이므로 행 번호가 곧 인덱스다.
    return {
        i: (int(r), bool(failed_by_run.get(int(r), False)))
        for i, r in enumerate(targets["run_id"].to_numpy())
        if r >= 0
    }


def evaluate(
    policy,
    dataset,
    preprocessor,
    *,
    root=None,
    batch_size: int = 32,
    num_workers: int = 4,
    max_batches: int | None = None,
    device=None,
) -> HeadMetrics:
    """held-out 평가 한 번. 학습 루프와 독립 스크립트가 모두 이걸 부른다."""
    preds = collect_predictions(
        policy, dataset, preprocessor,
        batch_size=batch_size, num_workers=num_workers,
        max_batches=max_batches, device=device,
    )
    lookup = None
    if root is not None:
        try:
            lookup = build_run_lookup(root)
        except Exception as exc:  # noqa: BLE001 - 평가 실패가 학습을 죽이면 안 된다
            logging.warning(f"run lookup 실패, 프레임 단위 지표만 낸다: {exc}")
    return compute_metrics(preds, lookup)
