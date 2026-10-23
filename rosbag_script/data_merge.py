from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset

repo_ids = [
    "moonjongsul/kcare_pi05_dataset_open_drawer_first_floor_left_merged",
    "moonjongsul/kcare_pi05_dataset_open_drawer_second_floor_left_merged",
    "moonjongsul/kcare_pi05_dataset_open_drawer_third_floor_left_merged",
]

datasets = [LeRobotDataset(rid) for rid in repo_ids]

merged = merge_datasets(
    datasets,
    output_repo_id="moonjongsul/kcare_pi05_dataset_open_drawer_all_left_merged",
)
merged.push_to_hub()