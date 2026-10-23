"""소스 데이터셋의 meta/episodes self-reference 인덱스 불일치 복구 스크립트.

문제:
    이 데이터셋들(이름이 ..._merged)은 물리적으로 meta/episodes/chunk-000/file-000.parquet
    하나에 모든 에피소드가 들어 있는데, 그 안의 `meta/episodes/file_index` 컬럼 값은
    [0,1,2,3...] 처럼 여러 파일에 나뉘어 있다고 잘못 기록돼 있다.

    - 정상 로딩(load_episodes)은 실제 존재하는 파일만 glob 하므로 문제없이 학습된다.
    - 하지만 aggregate_datasets(merge)는 이 file_index 컬럼을 그대로 믿고 없는
      file-001.parquet 등을 읽으려다 FileNotFoundError 로 실패한다.

복구:
    각 소스의 (유일한) 메타 parquet 안의
    `meta/episodes/chunk_index`, `meta/episodes/file_index` 컬럼을 실제 물리 배치인 0 으로 덮어쓴다.
    원본은 .bak 으로 백업한다.

    * 각 메타 파일 크기가 ~0.6MB 로 merge 시 파일 회전(100MB 임계) 이 일어나지 않으므로,
      전부 0 으로 맞추면 합쳐진 결과도 file-000 하나로 일관되게 생성된다.

사용법:
    python repair_source_meta.py
"""

import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO)

CACHE_ROOT = Path("/home/mjs/.cache/huggingface/lerobot/moonjongsul")

# 데이터 merge 에서 meta - 001.parquet 찾는 등 문제 발생 시 아래에 데이터셋 이름 작성해서 repair
REPO_DIR_NAMES = [
    "kcare_pi05_dataset_open_drawer_second_floor_right_merged",
    "kcare_pi05_dataset_open_drawer_third_floor_right_merged",
]

FIX_COLUMNS = ["meta/episodes/chunk_index", "meta/episodes/file_index"]


def repair_one(episodes_dir: Path):
    parquet_files = sorted(episodes_dir.rglob("*.parquet"))
    # .bak 은 제외
    parquet_files = [p for p in parquet_files if p.suffix == ".parquet"]

    if len(parquet_files) != 1:
        raise RuntimeError(
            f"예상과 다른 물리 파일 구성: {episodes_dir} 에 parquet {len(parquet_files)}개 "
            f"({[p.name for p in parquet_files]}). 전부 0 으로 맞추는 복구는 단일 파일 전제라 중단합니다."
        )

    path = parquet_files[0]
    expected = episodes_dir / "chunk-000" / "file-000.parquet"
    if path != expected:
        raise RuntimeError(f"유일 파일이 chunk-000/file-000 이 아님: {path}")

    table = pq.read_table(path)

    for col in FIX_COLUMNS:
        if col not in table.column_names:
            raise RuntimeError(f"{path} 에 컬럼 없음: {col}")
        i = table.column_names.index(col)
        field = table.schema.field(i)
        # 원본 dtype 유지하며 전부 0 으로 덮어쓴다
        zeros = pa.array([0] * table.num_rows, type=field.type)
        table = table.set_column(i, field, zeros)

    # 백업 후 덮어쓰기 (이미 .bak 있으면 유지)
    bak = path.with_suffix(".parquet.bak")
    if not bak.exists():
        bak.write_bytes(path.read_bytes())
        logging.info(f"백업 생성: {bak}")

    pq.write_table(table, path)
    logging.info(f"복구 완료: {path} ({table.num_rows} rows -> file_index/chunk_index=0)")


def main():
    for name in REPO_DIR_NAMES:
        episodes_dir = CACHE_ROOT / name / "meta" / "episodes"
        if not episodes_dir.exists():
            raise FileNotFoundError(f"경로 없음: {episodes_dir}")
        logging.info(f"복구 대상: {name}")
        repair_one(episodes_dir)
    logging.info("모든 소스 복구 완료. 이제 merge 를 다시 실행하세요.")


if __name__ == "__main__":
    main()
