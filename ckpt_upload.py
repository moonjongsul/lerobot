from pathlib import Path

from huggingface_hub import HfApi, create_repo


HF_USER = 'moonjongsul'
PROJECT = 'smolvla_open_drawer_a6000_b32x2_resume_260730'
REPO_ID = f'{HF_USER}/{PROJECT}'

LOCAL_PATH = f"/data/keti/mjs/lerobot/outputs/{PROJECT}/checkpoints"
CKPTS = [
    "best_step_0399400_loss_0.003570",
]
CKPTS = [f"{LOCAL_PATH}/{str(ckpt)}" for ckpt in CKPTS]


def main():
    api = HfApi()
    create_repo(REPO_ID, repo_type="model", exist_ok=True, private=False)

    for ckpt_dir in CKPTS:
        ckpt_path = Path(ckpt_dir)
        pretrained_dir = ckpt_path
        if not pretrained_dir.is_dir():
            raise FileNotFoundError(f"pretrained_model not found under {ckpt_path}")

        step_name = ckpt_path.name
        print(f"Uploading {pretrained_dir} -> {REPO_ID}:{step_name}/")
        api.upload_folder(
            repo_id=REPO_ID,
            repo_type="model",
            folder_path=str(pretrained_dir),
            path_in_repo=step_name,
            commit_message=f"Upload checkpoint step {step_name}",
        )

    print(f"Done. https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()

