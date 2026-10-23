from huggingface_hub import snapshot_download


LOCAL_DIR='/data/keti/mjs/lerobot/datasets'

DB_LIST = [
    'moonjongsul/kcare_pi05_dataset_open_drawer_first_floor_left_merged',
    'moonjongsul/kcare_pi05_dataset_open_drawer_second_floor_left_merged',
    'moonjongsul/kcare_pi05_dataset_open_drawer_third_floor_left_merged'
    
]

for d in DB_LIST:
    snapshot_download(
        repo_id=d,
        repo_type="dataset",
        # local_dir=LOCAL_DIR,
    )