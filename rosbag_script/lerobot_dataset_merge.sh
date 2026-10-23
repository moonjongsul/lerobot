# Merge train and validation splits back into one dataset
lerobot-edit-dataset \
    --new_repo_id moonjongsul/kcare_open_drawer_lr_mode_merged \
    --operation.type merge \
    --operation.repo_ids "[
        'moonjongsul/kcare_pi05_dataset_open_drawer_first_floor_left_merged',
        'moonjongsul/kcare_pi05_dataset_open_drawer_second_floor_left_merged',
        'moonjongsul/kcare_pi05_dataset_open_drawer_third_floor_left_merged',
        'moonjongsul/kcare_pi05_dataset_open_drawer_third_floor_right_merged',
        'moonjongsul/kcare_pi05_dataset_open_drawer_second_floor_right_merged'
    ]"

"""
(lerobot) mjs@ketiserver:~/workspace/lerobot/rosbag_script$ ./lerobot_dataset_merge.sh 
INFO 2026-07-28 10:08:53 _dataset.py:367 Loading 5 datasets to merge
INFO 2026-07-28 10:08:54 eo_utils.py:108 Using video codec: libsvtav1
INFO 2026-07-28 10:08:54 eo_utils.py:108 Using video codec: libsvtav1
INFO 2026-07-28 10:08:55 eo_utils.py:108 Using video codec: libsvtav1
INFO 2026-07-28 10:08:56 eo_utils.py:108 Using video codec: libsvtav1
INFO 2026-07-28 10:08:56 eo_utils.py:108 Using video codec: libsvtav1
INFO 2026-07-28 10:08:57 _dataset.py:372 Merging datasets into moonjongsul/kcare_open_drawer_lr_mode_merged
INFO 2026-07-28 10:08:57 ggregate.py:261 Start aggregate_datasets
Validate all meta data: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 37583.37it/s]
INFO 2026-07-28 10:08:59 ggregate.py:292 Find all tasks
Copy data and videos: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:37<00:00,  7.47s/it]
INFO 2026-07-28 10:09:36 ggregate.py:639 write tasks
INFO 2026-07-28 10:09:36 ggregate.py:642 write info
INFO 2026-07-28 10:09:36 ggregate.py:653 write stats
INFO 2026-07-28 10:09:36 ggregate.py:320 Aggregation complete.
INFO 2026-07-28 10:09:36 eo_utils.py:108 Using video codec: libsvtav1
INFO 2026-07-28 10:09:38 _dataset.py:379 Merged dataset saved to /home/mjs/.cache/huggingface/lerobot/moonjongsul/kcare_open_drawer_lr_mode_merged
INFO 2026-07-28 10:09:38 _dataset.py:380 Episodes: 1605, Frames: 260618
"""