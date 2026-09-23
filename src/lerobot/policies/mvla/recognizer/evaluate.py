"""Evaluation protocol for the recognizer.

Three rules, each of which caught a real error during development:

1. **Contiguous-block cross-validation.** Episode index is recording order,
   so contiguous blocks approximate separate sessions. Shuffling episodes
   into random folds lets a model that memorised lighting and tray-fill
   score 98% and then collapse to 45% -- below chance -- on a genuinely
   unseen session. `block_cv` never permutes; `random_cv` is kept only to
   show the gap.

2. **A placebo control.** The same pipeline run on a crop that cannot
   contain the object. It should score near chance. When it does not, the
   headline number is measuring the room. This is what exposed a 94.9%
   "mat state classifier" that was reading the workbench.

3. **Scoring by run, not by episode.** 32 episodes hold both a failed place
   and its successful retry, so per-episode aggregation averages a 0 and a
   1 into a label that does not exist.

`evaluate_task` reports the real crop and its placebo side by side, because
the real number is only interpretable next to the control.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class TaskResult:
    task: str
    n_units: int
    n_positive: int
    accuracy: float
    auc: float
    average_precision: float
    placebo_accuracy: float | None = None
    placebo_auc: float | None = None
    operating_points: pd.DataFrame | None = None
    per_class: dict[str, float] = field(default_factory=dict)

    @property
    def placebo_margin(self) -> float | None:
        """How much of the score the object itself accounts for."""
        if self.placebo_auc is None:
            return None
        return self.auc - self.placebo_auc

    def summary(self) -> str:
        lines = [
            f"{self.task}: {self.n_units} units ({self.n_positive} positive)",
            f"  accuracy {self.accuracy * 100:.1f}%   AUC {self.auc:.3f}   AP {self.average_precision:.3f}",
        ]
        if self.placebo_auc is not None:
            verdict = "ok" if self.placebo_margin > 0.15 else "SUSPECT"
            lines.append(
                f"  placebo  {self.placebo_accuracy * 100:.1f}%   AUC {self.placebo_auc:.3f}"
                f"   margin {self.placebo_margin:+.3f}  [{verdict}]"
            )
        if self.per_class:
            for name, value in sorted(self.per_class.items()):
                lines.append(f"    {name:14s} {value * 100:5.1f}%")
        return "\n".join(lines)


def _make_probe(probe_c: float):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=4000, C=probe_c, class_weight="balanced"),
    )


def block_cv_predict(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    n_blocks: int = 5,
    probe_c: float = 0.1,
) -> np.ndarray:
    """Out-of-fold probabilities using contiguous blocks of `groups`.

    `groups` is the episode index; blocks are formed by splitting the sorted
    unique episodes, never by shuffling them.
    """
    order = np.array(sorted(set(groups.tolist())))
    classes = sorted(set(labels.tolist()))
    proba = np.full((len(labels), len(classes)), np.nan)
    for block in np.array_split(order, n_blocks):
        test = np.isin(groups, block)
        train = ~test
        if len(set(labels[train].tolist())) < 2 or not test.any():
            continue
        probe = _make_probe(probe_c).fit(features[train], labels[train])
        raw = probe.predict_proba(features[test])
        for j, name in enumerate(probe.classes_):
            proba[test, classes.index(name)] = raw[:, j]
    return proba


def random_cv_predict(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    probe_c: float = 0.1,
    seed: int = 0,
) -> np.ndarray:
    """Grouped but shuffled folds. Optimistic -- reported for contrast only."""
    from sklearn.model_selection import StratifiedGroupKFold

    classes = sorted(set(labels.tolist()))
    proba = np.full((len(labels), len(classes)), np.nan)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train, test in cv.split(features, labels, groups):
        if len(set(labels[train].tolist())) < 2:
            continue
        probe = _make_probe(probe_c).fit(features[train], labels[train])
        raw = probe.predict_proba(features[test])
        for j, name in enumerate(probe.classes_):
            proba[test, classes.index(name)] = raw[:, j]
    return proba


def aggregate_by_unit(
    units: np.ndarray, labels: np.ndarray, proba: np.ndarray, classes: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Average frame probabilities within a run; return (labels, probs)."""
    table = pd.DataFrame(proba, columns=classes)
    table["unit"] = units
    table["label"] = labels
    grouped = table.groupby("unit")
    truth = grouped["label"].first().to_numpy()
    probs = grouped[classes].mean().to_numpy()
    return truth, probs


def operating_table(truth_binary: np.ndarray, score: np.ndarray) -> pd.DataFrame:
    """Precision / recall across thresholds, for choosing a gate setting."""
    rows = []
    for threshold in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1):
        flagged = score > threshold
        tp = int((flagged & (truth_binary == 1)).sum())
        fp = int((flagged & (truth_binary == 0)).sum())
        fn = int((~flagged & (truth_binary == 1)).sum())
        rows.append(
            {
                "threshold": threshold,
                "flagged": tp + fp,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": tp / max(tp + fp, 1),
                "recall": tp / max(tp + fn, 1),
            }
        )
    return pd.DataFrame(rows)


def evaluate_task(
    task: str,
    features: np.ndarray,
    labels: np.ndarray,
    units: np.ndarray,
    groups: np.ndarray,
    positive: str | None = None,
    placebo_features: np.ndarray | None = None,
    n_blocks: int = 5,
    probe_c: float = 0.1,
) -> TaskResult:
    """Contiguous-block evaluation of one task, with its placebo control."""
    classes = sorted(set(labels.tolist()))
    proba = block_cv_predict(features, labels, groups, n_blocks, probe_c)
    keep = ~np.isnan(proba).any(axis=1)
    truth, probs = aggregate_by_unit(units[keep], labels[keep], proba[keep], classes)
    predicted = np.array(classes)[probs.argmax(axis=1)]
    accuracy = accuracy_score(truth, predicted)

    per_class = {
        name: float((predicted[truth == name] == name).mean())
        for name in classes
        if (truth == name).any()
    }

    if positive is None:
        positive = classes[-1]
    binary = (truth == positive).astype(int)
    score = probs[:, classes.index(positive)]
    if len(set(binary.tolist())) > 1:
        auc = roc_auc_score(binary, score)
        ap = average_precision_score(binary, score)
        table = operating_table(binary, score)
    else:
        auc, ap, table = float("nan"), float("nan"), None

    placebo_accuracy = placebo_auc = None
    if placebo_features is not None:
        p_proba = block_cv_predict(placebo_features, labels, groups, n_blocks, probe_c)
        p_keep = ~np.isnan(p_proba).any(axis=1)
        p_truth, p_probs = aggregate_by_unit(
            units[p_keep], labels[p_keep], p_proba[p_keep], classes
        )
        placebo_accuracy = accuracy_score(
            p_truth, np.array(classes)[p_probs.argmax(axis=1)]
        )
        p_binary = (p_truth == positive).astype(int)
        if len(set(p_binary.tolist())) > 1:
            placebo_auc = roc_auc_score(p_binary, p_probs[:, classes.index(positive)])

    return TaskResult(
        task=task,
        n_units=len(truth),
        n_positive=int(binary.sum()),
        accuracy=accuracy,
        auc=auc,
        average_precision=ap,
        placebo_accuracy=placebo_accuracy,
        placebo_auc=placebo_auc,
        operating_points=table,
        per_class=per_class,
    )
