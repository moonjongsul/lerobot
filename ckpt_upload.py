from pathlib import Path

from huggingface_hub import HfApi, create_repo


HF_USER = 'moonjongsul'
PROJECT = 'manufacturing_kitting_smolvla_rot6d_260504'
REPO_ID = f'{HF_USER}/{PROJECT}'

LOCAL_PATH = "/data/keti/mjs/lerobot/outputs/smolvla_kitting_rot6d_a6000_b24x3_260504/checkpoints"
CKPTS = [
    "best_step_081500_loss_0.003167",
    "best_step_079200_loss_0.003667",
    "best_step_065400_loss_0.004289",
    "best_step_056200_loss_0.005239",
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