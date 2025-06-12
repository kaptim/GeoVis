import os
import zipfile
from huggingface_hub import snapshot_download

DATA_DIR = r"/home/kaptim/eth/mlmc/project/bottom_up/data/osv5m"


def download_data():
    """Download data from huggingface. This might take a while."""
    snapshot_download(repo_id="osv5m/osv5m", local_dir=DATA_DIR, repo_type="dataset")


def extract_data():
    """Extract downloaded data from .zip files. This might take a while."""
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith(".zip"):
                with zipfile.ZipFile(os.path.join(root, file), "r") as zip_ref:
                    zip_ref.extractall(root)
                os.remove(os.path.join(root, file))


def main():
    download_data()
    extract_data()


if __name__ == "__main__":
    main()
