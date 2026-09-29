"""체크포인트 하나를 held-out 에피소드로 평가한다.

    python -m lerobot.policies.mvla.eval_heads_cli \
        --checkpoint checkpoints/outputs/<job>/checkpoints/050000 \
        --split split_260923.json

세 가지를 잰다. 학습 루프 안의 `val/*`로는 셋 다 알 수 없다.

1. **헤드, 프롬프트별 (개루프).** 학습 프롬프트에는 정답 subtask와 런의
   `Mistake:` 플래그가 들어 있고 헤드는 프롬프트를 읽는다. 그래서 같은
   체크포인트를 `train` / `oracle` / `blind` 세 프롬프트로 재서, 헤드가
   이미지에서 읽는 몫과 프롬프트에서 베끼는 몫을 가른다.

2. **헤드, 폐루프.** 런타임에서 프롬프트의 subtask와 경과시간은 헤드 자신의
   이전 예측에서 온다. 그 되먹임을 그대로 재현해, 추적한 subtask가 정답
   경계를 얼마나 늦게 따라가는지, 아예 못 넘어가는지를 잰다.
   `--head-prompt self`는 정책과 같은 프롬프트로 헤드를 읽고, `blind`는
   subtask 없는 프롬프트로 읽는다. 둘 다 돈다.

3. **행동 (`--actions`).** held-out 액션 청크 오차. SmolVLA 체크포인트에도
   돌아가므로 두 런을 같은 기준으로 비교하는 유일한 수단이다. SmolVLA는
   원문 프롬프트(`recorded`, 방법을 알려준다)와 중립 프롬프트(`neutral`)
   둘 다로 재서, 베이스라인이 프롬프트 누수에 얼마나 기대는지도 본다.

결과는 표로 찍고 `<checkpoint>/eval/<이름>.json`에 남긴다.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from lerobot.utils.constants import ACTION

logger = logging.getLogger("mvla.eval")

VALUE_KEYS = ("value_subtask_pred", "value_subtask_spread", "value_episode_pred", "status_p_failure")


# ────────────────────────────────────────────────────────────── loading
def load_checkpoint(checkpoint: str | Path, device: str):
    """(train config, policy, preprocessor) -- 학습 때와 같은 구성으로."""
    import lerobot.policies  # noqa: F401 - mvla 등 config 서브클래스 등록
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    path = Path(checkpoint)
    if (path / "pretrained_model").is_dir():
        path = path / "pretrained_model"
    cfg = TrainPipelineConfig.from_pretrained(path)
    policy_cls = get_policy_class(cfg.policy.type)
    policy = policy_cls.from_pretrained(path)
    policy.to(device).eval()
    pre, _post = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(path),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return cfg, policy, pre


def make_val_dataset(cfg, episodes: list[int], prompt_mode: str = "blind"):
    """학습 루프가 val을 만드는 방식 그대로: 같은 factory, 증강만 끈다."""
    from lerobot.datasets.factory import make_dataset

    val_cfg = copy.deepcopy(cfg)
    val_cfg.dataset.episodes = list(episodes)
    val_cfg.dataset.image_transforms.enable = False
    ds = make_dataset(val_cfg)
    if hasattr(ds, "prompt_mode"):
        ds.prompt_mode = prompt_mode
    return ds


def _autocast(device: str):
    if device.startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def _strided(ds, stride: int):
    return ds if stride <= 1 else Subset(ds, range(0, len(ds), stride))


# ──────────────────────────────────────────────────── 1. open-loop heads
def eval_heads_open_loop(policy, pre, ds, root, *, stride, batch_size, num_workers, device):
    from .eval_heads import collect_predictions, compute_metrics, build_run_lookup

    with _autocast(device):
        preds = collect_predictions(
            policy, _strided(ds, stride), pre,
            batch_size=batch_size, num_workers=num_workers, device=torch.device(device),
        )
    return compute_metrics(preds, build_run_lookup(root))


# ─────────────────────────────────────────────────── 2. closed-loop heads
class _Tracker:
    """한 에피소드의 런타임 subtask 추적 상태.

    런타임과 같은 규칙: 헤드가 현재와 다른 subtask를 `hysteresis`번 연속
    예측하면 넘어가고, 그 순간 경과시간을 0으로 되돌린다. 시작 subtask는
    에피소드의 첫 정답 subtask다 -- 런타임에서 인식기+플래너가 정하는 몫이고,
    여기서 재려는 건 그 뒤의 추적이다.
    """

    def __init__(self, first_subtask: int, hysteresis: int):
        self.subtask = first_subtask
        self.t0 = 0.0
        self.hysteresis = hysteresis
        self._pending = (-1, 0)

    def update(self, pred: int, t: float) -> None:
        if pred < 0 or pred >= 6 or pred == self.subtask:
            self._pending = (-1, 0)
            return
        last, count = self._pending
        count = count + 1 if pred == last else 1
        self._pending = (pred, count)
        if count >= self.hysteresis:
            self.subtask, self.t0 = pred, t
            self._pending = (-1, 0)


def eval_heads_closed_loop(
    policy, pre, ds, *, stride, num_workers, device, hysteresis=2, head_prompts=("self", "blind")
):
    """헤드 예측이 자기 프롬프트와 경과시간을 결정하는 되먹임을 재현한다.

    관측(이미지·state)은 예측과 무관하므로 DataLoader가 미리 읽고, 프롬프트와
    경과시간만 메인 루프에서 추적 상태로 덮어쓴다. 에피소드들을 한 배치에
    나란히 넣고 시간축으로 한 스텝씩 전진한다.
    """
    from .modeling_mvla import ELAPSED_KEY

    wrapper = ds
    base = wrapper.dataset
    fps = float(base.fps)
    cfg = policy.config
    builder = wrapper.builder

    # 데이터셋 위치 -> (에피소드, 전역 인덱스)
    ep_col = np.asarray(base.hf_dataset["episode_index"])
    idx_col = np.asarray(base.hf_dataset["index"])
    episodes = sorted(set(ep_col.tolist()))
    positions = {ep: np.flatnonzero(ep_col == ep)[::stride] for ep in episodes}
    n_steps = max(len(p) for p in positions.values())

    # 스텝 우선 순서로 펼친 위치 목록과, 스텝별 배치 경계
    schedule, batches = [], []
    for k in range(n_steps):
        active = [ep for ep in episodes if k < len(positions[ep])]
        batches.append(list(range(len(schedule), len(schedule) + len(active))))
        schedule.extend((ep, int(positions[ep][k])) for ep in active)

    loader = DataLoader(
        Subset(wrapper, [pos for _, pos in schedule]),
        batch_sampler=batches,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
    )

    def first_label(ep):
        rows = idx_col[positions[ep]]
        labels = wrapper._subtask[rows]
        labelled = labels[labels >= 0]
        return int(labelled[0]) if len(labelled) else 0

    trackers = {hp: {ep: _Tracker(first_label(ep), hysteresis) for ep in episodes} for hp in head_prompts}
    out = {hp: {k: [] for k in ("index", "episode", "t", "target", "tracked", "pred", *VALUE_KEYS)}
           for hp in head_prompts}

    start = time.time()
    for step, batch in enumerate(loader):
        eps = [schedule[i][0] for i in batches[step]]
        rows = batch["index"].numpy().astype(int)
        t = (batch["frame_index"].numpy().astype(float)) / fps
        for hp in head_prompts:
            tr = [trackers[hp][ep] for ep in eps]
            neutral = [wrapper._neutral.get(ep, "") for ep in eps]
            if hp == "self":
                tasks = [
                    builder.build_inference(n, subtask=wrapper._subtask_sentence.get(x.subtask) or None,
                                            speed_bin_value=wrapper._speed.get(ep))
                    for n, x, ep in zip(neutral, tr, eps, strict=True)
                ]
            else:
                tasks = [builder.build_unconditional(n) for n in neutral]
            elapsed = torch.tensor(
                [(ti - x.t0) / cfg.elapsed_scale for ti, x in zip(t, tr, strict=True)], dtype=torch.float32
            )
            b = dict(batch)
            b["task"] = tasks
            b[ELAPSED_KEY] = elapsed
            with torch.no_grad(), _autocast(device):
                heads = policy.head_outputs(pre(b))
            pred = heads["subtask_logits"].argmax(-1).cpu().numpy()
            vs = policy.heads.value_subtask
            vals = {
                "value_subtask_pred": vs.expected_value(heads["value_subtask_logits"]),
                "value_subtask_spread": vs.spread(heads["value_subtask_logits"]),
                "value_episode_pred": policy.heads.value_episode.expected_value(heads["value_episode_logits"]),
                "status_p_failure": heads["status_logits"].softmax(-1)[:, 2],
            }
            o = out[hp]
            for i, (x, ti) in enumerate(zip(tr, t, strict=True)):
                o["tracked"].append(x.subtask)  # 이 프레임의 프롬프트가 쓴 subtask
                x.update(int(pred[i]), float(ti))
            o["index"].extend(rows.tolist())
            o["episode"].extend(eps)
            o["t"].extend(t.tolist())
            o["target"].extend(wrapper._subtask[rows].tolist())
            o["pred"].extend(pred.tolist())
            for k, v in vals.items():
                o[k].extend(v.float().cpu().numpy().tolist())
        if step % 100 == 0:
            logger.info(f"closed-loop step {step}/{n_steps} ({time.time() - start:.0f}s)")

    return {hp: {k: np.asarray(v) for k, v in o.items()} for hp, o in out.items()}


def closed_loop_metrics(rec: dict[str, np.ndarray], root) -> dict:
    """추적 정확도와 경계 지연, 그리고 폐루프 value로 돌린 정체 경보."""
    from .eval_heads import build_run_lookup, compute_metrics

    labelled = rec["target"] >= 0
    res = {
        "tracked_acc": float((rec["tracked"][labelled] == rec["target"][labelled]).mean()),
        "head_acc": float((rec["pred"][labelled] == rec["target"][labelled]).mean()),
    }

    delays, missed, spurious, n_eps_stuck = [], 0, 0, 0
    n_boundaries = 0
    for ep in np.unique(rec["episode"]):
        sel = rec["episode"] == ep
        order = np.argsort(rec["t"][sel])
        t, gt, tr = rec["t"][sel][order], rec["target"][sel][order], rec["tracked"][sel][order]
        # 정답 경계: 라벨 있는 프레임 사이에서 subtask가 바뀌는 지점
        lab = np.flatnonzero(gt >= 0)
        bounds = [lab[i] for i in range(1, len(lab)) if gt[lab[i]] != gt[lab[i - 1]]]
        n_boundaries += len(bounds)
        ends = bounds[1:] + [len(t)]
        for b, e in zip(bounds, ends, strict=True):
            hit = np.flatnonzero(tr[b:e] == gt[b])
            if len(hit):
                delays.append(float(t[b + hit[0]] - t[b]))
            else:
                missed += 1
        # 헛넘김: 추적 subtask가 바뀌었는데 그 값이 그 시점 정답이 아닌 경우
        changes = np.flatnonzero(np.diff(tr) != 0) + 1
        spurious += int(sum(1 for c in changes if gt[c] >= 0 and tr[c] != gt[c]))
        if len(bounds) and len(changes) == 0:
            n_eps_stuck += 1

    res.update({
        "boundaries": n_boundaries,
        "boundary_missed": missed / max(n_boundaries, 1),
        "boundary_delay_median_s": float(np.median(delays)) if delays else float("nan"),
        "boundary_delay_p90_s": float(np.percentile(delays, 90)) if delays else float("nan"),
        "spurious_switches": spurious,
        "episodes_never_switched": n_eps_stuck,
    })

    preds = {"index": rec["index"], "value_subtask_pred": rec["value_subtask_pred"],
             "status_p_failure": rec["status_p_failure"]}
    run_metrics = compute_metrics(preds, build_run_lookup(root))
    res.update({f"run/{k}": v for k, v in run_metrics.scalars.items()})
    return res


# ───────────────────────────────────────────────────────────── 3. actions
def eval_actions(policy, pre, ds, *, task_mode, stride, batch_size, num_workers, device, seed=0):
    """held-out 액션 청크 오차. 정규화 공간 MSE와 축별 RMSE(원 단위).

    `task_mode`: MVLA는 데이터셋 래퍼의 prompt_mode(oracle/blind)를 쓰고,
    SmolVLA는 recorded(원문) / neutral(방법을 뺀 문장)을 덮어쓴다.
    """
    from .data.derive import neutral_for_task

    loader = DataLoader(_strided(ds, stride), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=device.startswith("cuda"))
    sq_sum, sq_first, count, count_first = None, None, 0.0, 0.0
    gen = torch.Generator(device=device).manual_seed(seed)
    chunk = policy.config.chunk_size
    act_dim = policy.config.action_feature.shape[0]
    for batch in loader:
        if task_mode == "neutral":
            batch["task"] = [neutral_for_task(s) for s in batch["task"]]
        policy.reset()
        with torch.no_grad(), _autocast(device):
            b = pre(batch)
            noise = torch.randn(
                (len(batch["task"]), chunk, policy.config.max_action_dim), generator=gen, device=device
            )
            pred = policy.predict_action_chunk(b, noise=noise).float()
        gt = b[ACTION].float()
        pad = b.get(f"{ACTION}_is_pad")
        valid = (~pad).float() if pad is not None else torch.ones(gt.shape[:2], device=gt.device)
        n = min(pred.shape[1], gt.shape[1])
        err = (pred[:, :n, :act_dim] - gt[:, :n, :act_dim]) ** 2 * valid[:, :n, None]
        s = err.sum(dim=(0, 1)).cpu()
        s10 = err[:, :10].sum(dim=(0, 1)).cpu()
        sq_sum = s if sq_sum is None else sq_sum + s
        sq_first = s10 if sq_first is None else sq_first + s10
        count += float(valid[:, :n].sum())
        count_first += float(valid[:, :10].sum())

    mse_dim = (sq_sum / count).numpy()
    std = np.asarray(ds.meta.stats[ACTION]["std"], dtype=np.float64).reshape(-1)[:act_dim]
    names = (ds.meta.features[ACTION].get("names") or [str(i) for i in range(act_dim)])[:act_dim]
    return {
        "mse_norm": float(mse_dim.mean()),
        "mse_norm_first10": float((sq_first / count_first).numpy().mean()),
        "rmse": {n: float(np.sqrt(m) * s) for n, m, s in zip(names, mse_dim, std, strict=True)},
    }


# ──────────────────────────────────────────────────────────────── report
SUBTASK_SHORT = ["appr_flip", "flip", "appr_pick", "pick", "move", "place"]


def _fmt_confusion(conf: np.ndarray) -> str:
    rows = ["            " + " ".join(f"{s[:9]:>9}" for s in SUBTASK_SHORT) + "   recall"]
    for i, name in enumerate(SUBTASK_SHORT):
        total = conf[i].sum()
        rec = conf[i, i] / total if total else float("nan")
        rows.append(f"{name:>10}  " + " ".join(f"{v:9d}" for v in conf[i]) + f"   {rec:.3f}")
    return "\n".join(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="split_260923.json")
    p.add_argument("--modes", default="train,oracle,blind",
                   help="개루프 헤드 평가 프롬프트 (train,oracle,blind 중). 빈 문자열이면 생략")
    p.add_argument("--closed-loop", action="store_true", help="폐루프 subtask 추적 평가")
    p.add_argument("--actions", default="",
                   help="액션 평가 프롬프트. MVLA: oracle,blind / SmolVLA: recorded,neutral")
    p.add_argument("--stride", type=int, default=3, help="헤드 평가 프레임 간격 (3 = 10Hz)")
    p.add_argument("--action-stride", type=int, default=30, help="액션 평가 프레임 간격")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--name", default=None, help="결과 파일 이름 (기본: 평가 구성으로 만든다)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    val = json.loads(Path(args.split).read_text())["val"]
    cfg, policy, pre = load_checkpoint(args.checkpoint, args.device)
    is_mvla = hasattr(policy, "head_outputs")
    results: dict = {"checkpoint": str(args.checkpoint), "policy": cfg.policy.type, "val_episodes": len(val)}

    modes = [m for m in args.modes.split(",") if m] if is_mvla else []
    ds = None
    for mode in modes:
        ds = ds or make_val_dataset(cfg, val, mode)
        ds.prompt_mode = mode
        m = eval_heads_open_loop(policy, pre, ds, ds.root, stride=args.stride,
                                 batch_size=args.batch_size, num_workers=args.num_workers, device=args.device)
        results[f"heads/{mode}"] = {"scalars": m.scalars, "n_runs": m.n_runs, "n_failed_runs": m.n_failed_runs,
                                    "confusion": m.confusion.tolist() if m.confusion is not None else None}
        print(f"\n── heads [{mode}] ──  {m.summary()}")
        if m.confusion is not None:
            print(_fmt_confusion(m.confusion))

    if args.closed_loop and is_mvla:
        ds = ds or make_val_dataset(cfg, val, "blind")
        recs = eval_heads_closed_loop(policy, pre, ds, stride=args.stride,
                                      num_workers=args.num_workers, device=args.device)
        for hp, rec in recs.items():
            cl = closed_loop_metrics(rec, ds.root)
            results[f"closed_loop/{hp}"] = cl
            print(f"\n── closed loop [head prompt: {hp}] ──")
            for k, v in cl.items():
                print(f"  {k:<32} {v:.4f}" if isinstance(v, float) else f"  {k:<32} {v}")

    ds_a = None
    for mode in [m for m in args.actions.split(",") if m]:
        # MVLA는 래퍼의 prompt_mode로, SmolVLA는 배치의 task를 덮어써서 바꾼다.
        ds_a = ds_a or make_val_dataset(cfg, val, mode if is_mvla else "blind")
        if is_mvla:
            ds_a.prompt_mode = mode
        a = eval_actions(policy, pre, ds_a, task_mode=mode, stride=args.action_stride,
                         batch_size=args.batch_size, num_workers=args.num_workers, device=args.device)
        results[f"actions/{mode}"] = a
        rm = "  ".join(f"{k}:{v:.4f}" for k, v in a["rmse"].items())
        print(f"\n── actions [{mode}] ──  mse_norm {a['mse_norm']:.4f}  first10 {a['mse_norm_first10']:.4f}\n  rmse {rm}")

    name = args.name or "_".join(filter(None, [
        "heads-" + "-".join(modes) if modes else "",
        "closed" if args.closed_loop and is_mvla else "",
        "actions-" + args.actions.replace(",", "-") if args.actions else "",
    ]))
    out_dir = Path(args.checkpoint) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name or 'eval'}.json"
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
