"""메인 prompt의 작업이 끝났는지를 헤드가 판단할 수 있는가.

    python -m lerobot.policies.mvla.eval_termination \
        --checkpoint checkpoints/outputs/<job>/checkpoints/050000

두 가지를 잰다.

1. **종료 탐지.** 에피소드는 메인 prompt의 목표를 이룬 순간 끝난다(녹화 종료).
   그래서 `value_episode`의 타깃은 "이 prompt의 작업 끝까지 남은 비율"이고,
   0에 닿는 것이 곧 작업 종료다. 규칙 `value >= thr`가 `k`회 연속이면 종료로
   보고, 실제 끝 대비 언제 울리는지와 작업 중간에 잘못 울리는지를 잰다.
   비교용으로 현재 런타임 규칙(마지막 subtask에서 `value_subtask >= -0.05`)도 잰다.

2. **prompt 의존성.** 같은 장면이라도 목표가 다르면 판단이 달라야 한다.
     * kit 에피소드에서 pick이 끝난 순간(물체를 든 상태):
       pick prompt -> 끝, kit prompt -> 아직 (move, place가 남음)
     * pick 에피소드의 끝: pick prompt -> 끝, kit prompt -> 아직
   prompt의 목표 문장만 바꿔 같은 프레임을 두 번 읽는다.

프롬프트는 런타임과 같은 형태다: `Task: <중립 목표>. Subtask: <정답>. Quality: 5.
Mistake: false.` (Speed 생략 -- 런타임 기본값). 경과시간은 정답.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .data.derive import NEUTRAL_PROMPTS, TASK_GOALS
from .eval_heads_cli import _autocast, load_checkpoint, make_val_dataset

logger = logging.getLogger("mvla.termination")

GOAL_SENTENCE = {"pick": NEUTRAL_PROMPTS["pick"], "kit": NEUTRAL_PROMPTS["kit"]}
SWAP = {"pick": "kit", "kit": "pick"}
PICK_SUBTASK = 3  # meta/subtasks 순서: approach_flip, flip, approach_pick, pick, move, place


def episode_goal(task_sentence: str) -> str:
    goal = TASK_GOALS[task_sentence]
    return "kit" if goal == "place" else goal  # place 목표는 kit 문장으로 학습됐다


@torch.no_grad()
def read_values(policy, pre, ds, positions, goal_of_row, *, batch_size, num_workers, device):
    """주어진 위치의 프레임을 목표 prompt별로 읽는다. {goal: (value_episode, value_subtask)}."""
    builder = ds.builder
    loader = DataLoader(Subset(ds, positions), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=device.startswith("cuda"))
    out = {"own": ([], []), "swap": ([], []), "rows": []}
    for batch in loader:
        rows = batch["index"].numpy().astype(int)
        out["rows"].extend(rows.tolist())
        for which in ("own", "swap"):
            tasks = []
            for r in rows:
                goal = goal_of_row(r)
                if which == "swap":
                    goal = SWAP[goal]
                sub = ds._subtask_sentence.get(int(ds._subtask[r])) or None
                tasks.append(builder.build_inference(GOAL_SENTENCE[goal], subtask=sub, speed_bin_value=None))
            b = dict(batch)
            b["task"] = tasks
            with _autocast(device):
                h = policy.head_outputs(pre(b))
            ve = policy.heads.value_episode.expected_value(h["value_episode_logits"]).float().cpu().numpy()
            vs = policy.heads.value_subtask.expected_value(h["value_subtask_logits"]).float().cpu().numpy()
            out[which][0].extend(ve.tolist())
            out[which][1].extend(vs.tolist())
    return {k: (np.asarray(v[0]), np.asarray(v[1])) if k != "rows" else np.asarray(v) for k, v in out.items()}


def first_fire(v: np.ndarray, thr: float, k: int) -> int | None:
    run = 0
    for i, x in enumerate(v):
        run = run + 1 if x >= thr else 0
        if run >= k:
            return i
    return None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="split_260923.json")
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=12)
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    from .data.derive import episode_metadata
    from .data.segments import runs_frame

    val = json.loads(Path(args.split).read_text())["val"]
    cfg, policy, pre = load_checkpoint(args.checkpoint, args.device)
    ds = make_val_dataset(cfg, val, "oracle")
    base = ds.dataset
    fps = float(base.fps)
    meta = episode_metadata(ds.root).set_index("episode")
    runs = runs_frame(ds.root)

    ep_col = np.asarray(base.hf_dataset["episode_index"])
    idx_col = np.asarray(base.hf_dataset["index"])
    fr_col = np.asarray(base.hf_dataset["frame_index"])
    goal_by_ep = {ep: episode_goal(meta.loc[ep, "task"]) for ep in val}
    row_to_ep = dict(zip(idx_col.tolist(), ep_col.tolist(), strict=True))

    # 에피소드마다 stride 간격 + 마지막 프레임은 반드시 포함
    positions = []
    for ep in val:
        pos = np.flatnonzero(ep_col == ep)
        positions.extend(sorted(set(pos[:: args.stride].tolist()) | {int(pos[-1])}))

    vals = read_values(policy, pre, ds, positions, lambda r: goal_by_ep[row_to_ep[int(r)]],
                       batch_size=args.batch_size, num_workers=args.num_workers, device=args.device)
    rows = vals["rows"]
    pos_of_row = {int(r): i for i, r in enumerate(rows)}
    frame_of_row = dict(zip(idx_col.tolist(), fr_col.tolist(), strict=True))

    results: dict = {"checkpoint": args.checkpoint, "episodes": {}}

    # ── 1. 종료 탐지 ───────────────────────────────────────────────────
    per_ep = {}
    for ep in val:
        sel = np.array([row_to_ep[int(r)] == ep for r in rows])
        order = np.argsort([frame_of_row[int(r)] for r in rows[sel]])
        r_ep = rows[sel][order]
        t = np.array([frame_of_row[int(r)] for r in r_ep]) / fps
        ve, vs = vals["own"][0][sel][order], vals["own"][1][sel][order]
        ep_runs = runs[runs["episode"] == ep].sort_values("start")
        last = ep_runs.iloc[-1]
        per_ep[ep] = dict(t=t, ve=ve, vs=vs, t_end=t[-1], t_last_start=last["start"] / fps,
                          last_subtask=int(last["subtask"]), goal=goal_by_ep[ep],
                          # 목표를 못 이루고 끝난 에피소드만 뺀다. meta의 failed는
                          # "실패 런이 하나라도 있음"이라 복구 후 성공한 것까지 빠진다.
                          failed=bool(last["failed"]),
                          in_last=np.array([int(ds._subtask[int(r)]) == int(last["subtask"])
                                            and frame_of_row[int(r)] >= last["start"] for r in r_ep]))

    def evaluate_rule(signal: str, thr: float, k: int, gated: bool):
        fired, early, dt = 0, 0, []
        n = 0
        for ep, e in per_ep.items():
            if e["failed"]:
                continue  # 에피소드 단위 실패는 목표를 이루지 못하고 끝났다
            n += 1
            if signal == "or":
                # 마지막 subtask에서 value_subtask >= thr 또는 value_episode >= -0.05
                v = np.where((e["vs"] >= thr) | (e["ve"] >= -0.05), 0.0, -1.0)
                thr_eff = -0.5
            else:
                v = e[signal].copy()
                thr_eff = thr
            if gated:
                v = np.where(e["in_last"], v, -1.0)
            i = first_fire(v, thr_eff, k)
            if i is None:
                continue
            fired += 1
            dt.append(float(e["t"][i] - e["t_end"]))
            if e["t"][i] < e["t_last_start"]:
                early += 1
        dt = np.asarray(dt)
        return {
            "episodes": n, "fired": fired / max(n, 1), "fired_before_last_subtask": early / max(n, 1),
            "dt_median_s": float(np.median(dt)) if len(dt) else float("nan"),
            "dt_p10_s": float(np.percentile(dt, 10)) if len(dt) else float("nan"),
            "dt_p90_s": float(np.percentile(dt, 90)) if len(dt) else float("nan"),
        }

    table = []
    for signal in ("ve", "vs", "or"):
        for gated in (False, True):
            if signal in ("vs", "or") and not gated:
                continue  # value_subtask는 subtask마다 0에 닿으므로 게이트 없이는 무의미
            for thr in (-0.02, -0.05, -0.10, -0.15):
                for k in (1, 3):
                    r = evaluate_rule(signal, thr, k, gated)
                    r.update(signal={"ve": "value_episode", "vs": "value_subtask", "or": "vs|ve>=-.05"}[signal],
                             gated=gated, thr=thr, k=k)
                    table.append(r)
    results["termination"] = table

    # 마지막 프레임 값 분포 (목표별)
    last_vals = {}
    for goal in ("pick", "kit"):
        eps = [e for e in per_ep.values() if e["goal"] == goal and not e["failed"]]
        last_vals[goal] = {
            "n": len(eps),
            "value_episode_at_end": [float(e["ve"][-1]) for e in eps],
            "value_episode_max_before_last_subtask": [
                float(e["ve"][e["t"] < e["t_last_start"]].max()) if (e["t"] < e["t_last_start"]).any() else None
                for e in eps],
        }
    results["last_values"] = last_vals

    # ── 2. prompt 의존성 ───────────────────────────────────────────────
    probes = {"kit_ep@pick_end": {"own": [], "swap": []}, "pick_ep@end": {"own": [], "swap": []}}
    for ep, e in per_ep.items():
        if e["failed"]:
            continue
        if e["goal"] == "pick":
            i = pos_of_row[int(rows[[row_to_ep[int(r)] == ep for r in rows]].max())]
            probes["pick_ep@end"]["own"].append(float(vals["own"][0][i]))
            probes["pick_ep@end"]["swap"].append(float(vals["swap"][0][i]))
        else:
            # kit 에피소드 안의 마지막 성공 pick 런의 끝 프레임
            ep_runs = runs[(runs["episode"] == ep) & (runs["subtask"] == PICK_SUBTASK) & (~runs["failed"])]
            if ep_runs.empty:
                continue
            end_frame = int(ep_runs.iloc[-1]["end"])
            cand = [r for r in rows if row_to_ep[int(r)] == ep and frame_of_row[int(r)] <= end_frame]
            if not cand:
                continue
            r = max(cand, key=lambda x: frame_of_row[int(x)])
            i = pos_of_row[int(r)]
            probes["kit_ep@pick_end"]["own"].append(float(vals["own"][0][i]))
            probes["kit_ep@pick_end"]["swap"].append(float(vals["swap"][0][i]))
    results["prompt_probe"] = probes

    # ── 출력 ─────────────────────────────────────────────────────────
    print("\n── 종료 탐지 (성공 에피소드, 실제 끝 대비 발동 시각 dt: 음수 = 일찍) ──")
    print(f"{'signal':<14}{'gate':<6}{'thr':>6}{'k':>3}{'fired':>8}{'early':>8}{'dt_med':>8}{'p10':>8}{'p90':>8}")
    for r in table:
        print(f"{r['signal']:<14}{'last' if r['gated'] else '-':<6}{r['thr']:>6.2f}{r['k']:>3}"
              f"{r['fired']:>8.2f}{r['fired_before_last_subtask']:>8.2f}"
              f"{r['dt_median_s']:>8.2f}{r['dt_p10_s']:>8.2f}{r['dt_p90_s']:>8.2f}")
    print("\n── 마지막 프레임 value_episode ──")
    for goal, d in last_vals.items():
        v = np.asarray(d["value_episode_at_end"])
        m = np.asarray([x for x in d["value_episode_max_before_last_subtask"] if x is not None])
        print(f"  {goal:<5} n={d['n']:<3} at_end median {np.median(v):.3f} min {v.min():.3f}"
              f" | 마지막 subtask 이전 최대값 median {np.median(m):.3f} max {m.max():.3f}")
    print("\n── prompt 의존성: value_episode (own = 에피소드 목표, swap = 다른 목표) ──")
    for name, d in probes.items():
        own, swap = np.asarray(d["own"]), np.asarray(d["swap"])
        if len(own):
            print(f"  {name:<18} n={len(own):<3} own median {np.median(own):.3f}  swap median {np.median(swap):.3f}"
                  f"  own>swap {np.mean(own > swap):.2f}")

    out = Path(args.checkpoint) / "eval" / "termination.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1, default=float))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
