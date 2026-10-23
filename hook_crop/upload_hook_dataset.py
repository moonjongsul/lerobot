import sys
from pathlib import Path

from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).parent))
from build_hook_dataset import DST_ROOT  # noqa: E402

REPO_ID = "moonjongsul/kcare_hang_cup_red_black_rot6d_hookcrop"
CODEBASE_TAG = "v3.0"

api = HfApi()
api.create_repo(REPO_ID, repo_type="dataset", private=False, exist_ok=True)
print(f"[upload] {DST_ROOT} -> {REPO_ID}", flush=True)
commit = api.upload_folder(
    repo_id=REPO_ID,
    repo_type="dataset",
    folder_path=DST_ROOT,
    commit_message="Add SAM3 hook crop camera (observation.images.hook)",
)
api.create_tag(REPO_ID, repo_type="dataset", tag=CODEBASE_TAG, revision=commit.oid, exist_ok=True)
print(f"[done] revision {commit.oid}", flush=True)
