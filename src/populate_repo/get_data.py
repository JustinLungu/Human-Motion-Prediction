import os
import shutil
import zipfile
import yaml

CONFIG_PATH = "configs/config.yaml"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def run_cmd(cmd: str):
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"Command failed: {cmd}")


def ensure_empty_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def find_uci_root(extract_dir: str) -> str:
    for root, dirs, _ in os.walk(extract_dir):
        for d in dirs:
            if d == "UCI HAR Dataset":
                return os.path.join(root, d)
    raise FileNotFoundError("Could not find 'UCI HAR Dataset' after extraction.")


def sanity_check(uci_root: str):
    sample = os.path.join(
        uci_root, "train", "Inertial Signals", "body_acc_x_train.txt"
    )
    if not os.path.exists(sample):
        raise FileNotFoundError(f"Sanity check failed: {sample}")

    with open(sample, "r") as f:
        n_cols = len(f.readline().split())

    print("Sanity check passed:")
    print(f"  Dataset root: {uci_root}")
    print(f"  Columns per row (expected 128): {n_cols}")


def main():
    cfg = load_config()
    ds = cfg["dataset"]
    paths = cfg["paths"]

    file_id = ds["drive_file_id"]
    zip_name = ds["zip_name"]
    raw_dir = paths["raw_data_dir"]
    target_dir = paths["dataset_dir"]
    temp_dir = paths["temp_dir"]

    print("[1/6] Downloading dataset from Google Drive...")
    run_cmd(f"gdown --id {file_id} -O {zip_name}")

    print("[2/6] Extracting zip...")
    ensure_empty_dir(temp_dir)
    with zipfile.ZipFile(zip_name, "r") as zf:
        zf.extractall(temp_dir)

    print("[3/6] Locating dataset root...")
    uci_root = find_uci_root(temp_dir)

    print("[4/6] Moving dataset to final location...")
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    shutil.move(uci_root, target_dir)

    print("[5/6] Cleaning up...")
    shutil.rmtree(temp_dir)
    os.remove(zip_name)

    print("[6/6] Running sanity check...")
    sanity_check(target_dir)

    print("\nDataset ready at:")
    print(f"  {target_dir}")


if __name__ == "__main__":
    main()
