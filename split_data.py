import os
import shutil
import random

# Đường dẫn gốc
original_dataset_dir = "dogs-vs-cats/train"  # chứa toàn ảnh cat.* và dog.*
base_dir = "data"  # thư mục sẽ tạo train/val/test

# Tạo thư mục train/val/test/cats và train/val/test/dogs
for split in ["train", "val", "test"]:
    for category in ["cats", "dogs"]:
        os.makedirs(os.path.join(base_dir, split, category), exist_ok=True)

# Tỉ lệ chia
split_ratio = {"train": 0.7, "val": 0.15, "test": 0.15}

# Lấy danh sách file cat và dog
all_files = os.listdir(original_dataset_dir)
cat_files = [f for f in all_files if f.startswith("cat")]
dog_files = [f for f in all_files if f.startswith("dog")]

for category, files in zip(["cats", "dogs"], [cat_files, dog_files]):
    random.shuffle(files)
    n_total = len(files)
    n_train = int(n_total * split_ratio["train"])
    n_val = int(n_total * split_ratio["val"])
    n_test = n_total - n_train - n_val

    splits = {
        "train": files[:n_train],
        "val": files[n_train:n_train+n_val],
        "test": files[n_train+n_val:]
    }

    for split_name, split_files in splits.items():
        for fname in split_files:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(base_dir, split_name, category, fname)
            shutil.copyfile(src, dst)

print("✅ Done splitting cats and dogs!")