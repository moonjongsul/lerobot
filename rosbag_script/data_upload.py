# import sys
# sys.path.insert(0, "/workspace/m.ax/thirdparty/lerobot/src")
# from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ds = LeRobotDataset(
#     repo_id="moonjongsul/flip_object",
#     root="/workspace/m.ax/datasets/flip_object",
# )
# ds.push_to_hub(private=False, tags=["robotics", "franka_fr3"])

#!/usr/bin/env python3
"""
LeRobot 데이터셋 HuggingFace 업로드 스크립트

Usage:
    python upload_lerobot_dataset.py \
        --dataset-dir /workspace/datasets/lerobot/manufacturing_parts_kitting_dataset \
        --repo-id moonjongsul/manufacturing_parts_kitting_dataset \
        --private
"""

import argparse
from pathlib import Path
from huggingface_hub import HfApi, whoami

DB_LOCAL_PATH = "/home/mjs/.cache/huggingface/lerobot/moonjongsul/"
HF_USER = "moonjongsul"

UPLOAD = []
UPLOAD.append("kcare_open_drawer_left_mode_merged")
# UPLOAD.append("manufacturing_kitting_dataset_flip_object")
# UPLOAD.append("manufacturing_kitting_dataset_kit_object")


"""
for db_name in UPLOAD:
    dataset_dir = Path(DB_LOCAL_PATH) / db_name
    repo_id = f"{HF_USER}/{db_name}"`
    main(dataset_dir=dataset_dir, repo_id=repo_id, private=False)
"""



def main(dataset_dir=None, repo_id=None, private=False):
    # parser = argparse.ArgumentParser(description="LeRobot 데이터셋 HuggingFace 업로드")
    # parser.add_argument("--dataset-dir", default=Path("/workspace/m.ax/datasets/manufacturing_kitting_dataset_kit_object"), 
    #                     type=Path, help="로컬 LeRobot 데이터셋 경로")
    # parser.add_argument("--repo-id", default="moonjongsul/manufacturing_kitting_dataset_kit_object",
    #                     type=str, help="HuggingFace repo ID (예: moonjongsul/my_dataset)")
    # parser.add_argument("--private", default=False, action="store_true",
    #                     help="비공개 레포지토리로 업로드")
    # args = parser.parse_args()

    # 로그인 확인
    try:
        user = whoami()
        print(f"로그인 확인: {user['name']}")
    except Exception:
        print("❌ HuggingFace 로그인 필요: python -c \"from huggingface_hub import login; login()\"")
        return

    api = HfApi()

    # 기존 레포 삭제 후 재생성 (완전히 새로 올릴 때)
    api.delete_repo(repo_id=repo_id, repo_type="dataset", missing_ok=True)

    # 레포지토리 생성 (이미 있으면 skip)
    print(f"레포지토리 확인/생성: {repo_id}")
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        exist_ok=True,
        private=private,
    )

    # 데이터셋 폴더 전체 업로드
    print(f"업로드 중: {dataset_dir} → {repo_id}")
    api.upload_folder(
        folder_path=str(dataset_dir),
        repo_id=repo_id,
        repo_type="dataset",
    )
    
    print(f"태깅 ")
    api.create_tag(
        repo_id=repo_id,
        repo_type="dataset",
        tag="v3.0",
    )

    print(f"\n✅ 업로드 완료!")
    print(f"   https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    for db_name in UPLOAD:
        dataset_dir = Path(DB_LOCAL_PATH) / db_name
        repo_id = f"{HF_USER}/{db_name}"
        main(dataset_dir=dataset_dir, repo_id=repo_id, private=False)