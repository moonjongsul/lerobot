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


def main():
    parser = argparse.ArgumentParser(description="LeRobot 데이터셋 HuggingFace 업로드")
    parser.add_argument("--dataset-dir", default=Path("/workspace/datasets/lerobot/manufacturing_parts_kitting_dataset"), 
                        type=Path, help="로컬 LeRobot 데이터셋 경로")
    parser.add_argument("--repo-id", default="moonjongsul/manufacturing_parts_kitting_dataset",
                        type=str, help="HuggingFace repo ID (예: moonjongsul/my_dataset)")
    parser.add_argument("--private", default=False, action="store_true",
                        help="비공개 레포지토리로 업로드")
    args = parser.parse_args()

    # 로그인 확인
    try:
        user = whoami()
        print(f"로그인 확인: {user['name']}")
    except Exception:
        print("❌ HuggingFace 로그인 필요: python -c \"from huggingface_hub import login; login()\"")
        return

    api = HfApi()

    # 기존 레포 삭제 후 재생성 (완전히 새로 올릴 때)
    # api.delete_repo(repo_id=args.repo_id, repo_type="dataset", missing_ok=True)

    # 레포지토리 생성 (이미 있으면 skip)
    print(f"레포지토리 확인/생성: {args.repo_id}")
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        exist_ok=True,
        private=args.private,
    )

    # 데이터셋 폴더 전체 업로드
    print(f"업로드 중: {args.dataset_dir} → {args.repo_id}")
    api.upload_folder(
        folder_path=str(args.dataset_dir),
        repo_id=args.repo_id,
        repo_type="dataset",
    )

    print(f"\n✅ 업로드 완료!")
    print(f"   https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()