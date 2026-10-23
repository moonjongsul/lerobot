from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "moonjongsul/kcare_hang_cup_red_black_rot6d_aug_pi05_261020"
CKPT_DIR = Path(
    "/data/keti/mjs/lerobot/outputs/pi05_hang_cup_red_black_rot6d_400_aug_resume80k_b12x3_261020/checkpoints"
)
TOP_K = 3


def best_loss(name: str) -> float:
    return float(name.rsplit("_loss_", 1)[1])


CKPTS = sorted((p.name for p in CKPT_DIR.glob("best_step_*")), key=best_loss)[:TOP_K]

api = HfApi()
api.create_repo(REPO_ID, repo_type="model", private=False, exist_ok=True)

print("[top-k]", *CKPTS, sep="\n  ", flush=True)
for name in CKPTS:
    src = CKPT_DIR / name / "pretrained_model"
    print(f"[upload] {src} -> {REPO_ID}/{name}", flush=True)
    api.upload_folder(
        repo_id=REPO_ID,
        folder_path=src,
        path_in_repo=name,
        commit_message=f"Add checkpoint {name}",
    )
    print(f"[done] {name}", flush=True)

print("[all done]", flush=True)
