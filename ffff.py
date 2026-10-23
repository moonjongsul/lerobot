from huggingface_hub import list_repo_commits, hf_hub_download
import json

# for c in list_repo_commits("lerobot/pi05_base"):
#     p = hf_hub_download("lerobot/pi05_base", "policy_preprocessor.json", revision=c.commit_id)
#     has_rel = "relative_actions_processor" in open(p).read()
#     print(c.commit_id[:8], c.created_at, "relative:", has_rel)


for c in list_repo_commits("lerobot/pi05_base"):
    if c.commit_id.startswith("a538eb27"):
        print(c.commit_id)