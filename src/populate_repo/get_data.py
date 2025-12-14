# src/populate_repo/get_data.py

import os
import shutil
import zipfile
from typing import Optional

import yaml

CONFIG_PATH = "configs/config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def run_cmd(cmd: str) -> None:
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"Command failed: {cmd}")


def ensure_empty_dir(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def find_inner_zip(search_dir: str) -> Optional[str]:
    for root, _, files in os.walk(search_dir):
        for f in files:
            if f.lower().endswith(".zip"):
                return os.path.join(root, f)
    return None


def find_uci_root(extract_dir: str) -> str:
    """
    Find the actual UCI HAR dataset root by signature files and folders.
    Expected:
      train/
      test/
      activity_labels.txt
    """
    for root, dirs, files in os.walk(extract_dir):
        if "train" in dirs and "test" in dirs and "activity_labels.txt" in files:
            return root
    raise FileNotFoundError(
        "Could not locate dataset root. Expected a folder containing train/, test/, and activity_labels.txt."
    )


def sanity_check(uci_root: str) -> None:
    sample = os.path.join(uci_root, "train", "Inertial Signals", "body_acc_x_train.txt")
    if not os.path.exists(sample):
        raise FileNotFoundError(f"Sanity check failed, missing file: {sample}")

    with open(sample, "r") as f:
        first_line = f.readline().strip()

    n_cols = len(first_line.split())
    print("Sanity check passed.")
    print(f"  Dataset root: {uci_root}")
    print(f"  Example file: {sample}")
    print(f"  Columns per row (expected 128): {n_cols}")


def main() -> None:
    cfg = load_config()
    ds = cfg["dataset"]
    paths = cfg["paths"]

    file_id = ds["drive_file_id"]
    zip_name = ds.get("zip_name", "UCI_HAR_Dataset.zip")

    raw_dir = paths["raw_data_dir"]
    target_dir = paths["dataset_dir"]
    temp_dir = paths["temp_dir"]

    ensure_dir(raw_dir)

    zip_path = os.path.join(raw_dir, zip_name)

    print("[1/7] Downloading dataset from Google Drive...")
    run_cmd(f"gdown --id {file_id} -O \"{zip_path}\"")

    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Zip was not downloaded: {zip_path}")

    print("[2/7] Extracting outer zip...")
    ensure_empty_dir(temp_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(temp_dir)

    # Handle nested zip if the downloaded file is a wrapper archive
    inner_zip = find_inner_zip(temp_dir)
    if inner_zip is not None:
        print("[3/7] Found nested zip, extracting nested zip...")
        nested_dir = os.path.join(temp_dir, "_nested")
        ensure_empty_dir(nested_dir)
        with zipfile.ZipFile(inner_zip, "r") as zf:
            zf.extractall(nested_dir)
        search_root = nested_dir
    else:
        print("[3/7] No nested zip found.")
        search_root = temp_dir

    print("[4/7] Locating dataset root...")
    uci_root = find_uci_root(search_root)

    print("[5/7] Moving dataset to final location...")
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    shutil.move(uci_root, target_dir)

    print("[6/7] Cleaning up temporary files...")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    if os.path.exists(zip_path):
        os.remove(zip_path)

    print("[7/7] Running sanity check...")
    sanity_check(target_dir)

    print("\nDone. Dataset ready at:")
    print(f"  {target_dir}")


if __name__ == "__main__":
    main()
