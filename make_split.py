#!/usr/bin/env python
"""xarm7-kitting-260923의 train/val 에피소드 분할을 계산해 저장한다.

데이터셋을 복사하지 않는다. `--dataset.episodes`에 넘길 인덱스 목록만 만든다.
LeRobot의 `split_dataset`은 "비디오 구간 = 정확히 length 프레임"을 가정하는데
이 데이터셋은 292개 중 288개에서 비디오가 평균 2% 더 길어(녹화 종료 시 여분)
그 assert에 걸린다. 원본 학습에는 무해한 차이다 — 데이터셋이 timestamp로
프레임을 조회하므로 뒤쪽 여분은 읽히지 않는다.

분할 방식이 단순한 "뒤쪽 20%"가 아닌 이유:
에피소드가 task별 연속 블록으로 기록되어 있어(ep 0-40 flip계열, 41-120 kit,
121-151/265-291 place, 152-182 pick, 183-264 kit-with-move) 전체의 뒤쪽 20%를
자르면 검증셋에 task 2종만 들어가고 flip이 필요한 에피소드가 하나도 남지
않는다. MVLA가 측정하려는 능력이 바로 "이미지를 보고 flip 필요 여부를 판단"
하는 것이므로 그 분할로는 아무것도 측정할 수 없다.

대신 각 task 블록 안에서 뒤쪽 20%씩 떼어낸다. 블록 내부의 연속성이 유지되므로
인식기 개발 때 겪은 세션 누수(랜덤 CV에서 98.3%가 나왔지만 플라시보 대조에서
94.9%로 드러난 사고)를 피할 수 있다.

    python make_split.py            # 계산하고 split_260923.json에 저장
    python make_split.py --show     # 저장된 분할을 다시 출력
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.mvla.data.segments import (
    episodes_frame,
    runs_frame,
    subtask_names,
)

SRC_REPO_ID = "moonjongsul/xarm7-kitting-260923"
VAL_FRACTION = 0.2
OUT_PATH = Path(__file__).parent / "split_260923.json"


def compute_split(root: Path) -> dict[str, list[int]]:
    """task 블록별로 뒤쪽 VAL_FRACTION을 검증셋으로 뗀다."""
    episodes = episodes_frame(root).sort_values("episode")

    val: list[int] = []
    for _task, group in episodes.groupby("task", sort=False):
        group = group.sort_values("episode")
        n_val = max(1, round(len(group) * VAL_FRACTION))
        val += [int(e) for e in group["episode"].tail(n_val)]

    val_set = set(val)
    train = [int(e) for e in episodes["episode"] if e not in val_set]
    return {"train": sorted(train), "val": sorted(val)}


def describe(root: Path, split: dict[str, list[int]]) -> None:
    """분할이 실제로 쓸 만한지 — task와 flip이 양쪽에 고루 들어갔는지 — 보고한다."""
    episodes = episodes_frame(root)
    runs = runs_frame(root)
    names = subtask_names(root)

    flip = runs.assign(f=runs["subtask"].map(names).eq("flip")).groupby("episode")["f"].any()
    fail = runs.groupby("episode")["failed"].any()
    episodes = episodes.join(flip.rename("needs_flip"), on="episode").join(
        fail.rename("has_fail"), on="episode"
    )
    episodes[["needs_flip", "has_fail"]] = episodes[["needs_flip", "has_fail"]].fillna(False)

    for name, eps in split.items():
        part = episodes[episodes["episode"].isin(eps)]
        print(
            f"  {name:5s} {len(part):3d}ep  "
            f"task {part['task'].nunique()}종  "
            f"flip필요 {int(part['needs_flip'].sum()):3d}  "
            f"실패포함 {int(part['has_fail'].sum()):2d}"
        )

    missing = set(episodes["task"]) - set(episodes[episodes["episode"].isin(split["val"])]["task"])
    if missing:
        raise SystemExit(f"val에 빠진 task가 있다: {missing}")


def as_cli_arg(episodes: list[int]) -> str:
    """`--dataset.episodes`가 받는 형식."""
    return "[" + ",".join(str(e) for e in episodes) + "]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=SRC_REPO_ID)
    parser.add_argument("--show", action="store_true", help="저장된 분할을 출력만 한다")
    args = parser.parse_args()

    if args.show:
        split = json.loads(OUT_PATH.read_text())
    else:
        meta = LeRobotDatasetMetadata(args.repo_id)
        split = compute_split(meta.root)
        OUT_PATH.write_text(json.dumps(split, indent=2))

    meta = LeRobotDatasetMetadata(args.repo_id)
    print(f"{args.repo_id}  총 {meta.total_episodes}개")
    describe(meta.root, split)
    print(f"\n저장: {OUT_PATH}")
    print("\n학습 스크립트에 넣을 값:")
    print(f"  TRAIN_EPISODES='{as_cli_arg(split['train'])}'")
    print(f"\nval 에피소드 ({len(split['val'])}개):")
    print(f"  {as_cli_arg(split['val'])}")


if __name__ == "__main__":
    main()
