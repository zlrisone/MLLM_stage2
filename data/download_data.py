# from huggingface_hub import snapshot_download

# local_dir = snapshot_download(
#     repo_id="liuhaotian/LLaVA-CC3M-Pretrain-595K",
#     repo_type="dataset",
#     local_dir="./data/llava_cc3m_raw",
#     local_dir_use_symlinks=False,
# )
# print(local_dir)

# import zipfile
# import os

# zip_path = "./data/llava_cc3m_raw/images.zip"
# image_root = "./data/llava_cc3m_raw/images"

# os.makedirs(image_root, exist_ok=True)

# with zipfile.ZipFile(zip_path, "r") as zf:
#     zf.extractall(image_root)

# print("done")

import os

target = "GCC_train_000572859.jpg"
found = []
image_root = "./data/llava_cc3m_raw"
for root, dirs, files in os.walk(image_root):
    if target in files:
        found.append(os.path.join(root, target))

print(found[:5])

import json

with open("./data/llava_cc3m_raw/chat.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(type(data), len(data))
print(data[0].keys())
print(data[0]["image"])
print(data[0]["conversations"])