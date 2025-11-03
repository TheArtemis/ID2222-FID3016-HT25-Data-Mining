from pathlib import Path
import os
import shutil

# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
os.makedirs(DATA_DIR, exist_ok=True)

# MUST set this before importing kagglehub so it uses the project data dir
os.environ["KAGGLEHUB_CACHE_DIR"] = str(DATA_DIR)

import kagglehub

# Download dataset (returns path to extracted files)
dataset_id = "shubchat/1002-short-stories-from-project-guttenberg"
downloaded_path = Path(kagglehub.dataset_download(dataset_id))

print("kagglehub returned:", downloaded_path)
print("Expected data dir:", DATA_DIR)

# If kagglehub still wrote elsewhere, copy contents into DATA_DIR
if not str(downloaded_path).startswith(str(DATA_DIR)):
    for item in downloaded_path.iterdir():
        dest = DATA_DIR / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

print("Final files are in:", DATA_DIR)
