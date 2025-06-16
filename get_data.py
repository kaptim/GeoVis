import os
import zipfile
import pandas as pd
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


def shuffle_train_csv():
    """Shuffle the rows of train.csv. After downloading, the rows
    are grouped by category, i.e., one country after another, which
    may have adverse effects on classification tasks"""
    train = pd.read_csv(DATA_DIR + "/train.csv")
    train = train.sample(frac=1).reset_index(drop=True)
    train.to_csv(DATA_DIR + "/train.csv", index=False)


def main():
    download_data()
    extract_data()
    shuffle_train_csv()


if __name__ == "__main__":
    main()
