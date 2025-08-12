import os
import shutil

import kagglehub

DATASET_NAME = "olistbr/brazilian-ecommerce"
LOCAL_DATASET_DIR = "/srv/data/dataset"


def download():
    """Download dataset using kagglehub and copy the archive to the local dataset directory."""
    print("Downloading dataset from kagglehub...")
    path = kagglehub.dataset_download(DATASET_NAME)
    print("Path to dataset files:", path)

    os.makedirs(LOCAL_DATASET_DIR, exist_ok=True)
    for filename in os.listdir(path):
        full_src_path = os.path.join(path, filename)
        full_dst_path = os.path.join(LOCAL_DATASET_DIR, filename)
        shutil.copy(full_src_path, full_dst_path)
        print(f"Copied {filename} to {full_dst_path}")


if __name__ == "__main__":
    download()
