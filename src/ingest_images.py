import os
from pathlib import Path
from time import time

from azure.storage.blob import BlobServiceClient


# --------- CONFIG: EDIT ONLY IF YOUR PATHS CHANGE ---------

# Local folder on your laptop that contains "yes" and "no"
LOCAL_DATASET_DIR = Path(
    r"C:\Users\MyAdmin\Desktop\Datalab5\assignment\data\brain_tumor_dataset"
)

# Azure Storage settings
STORAGE_ACCOUNT_NAME = os.environ.get("STORAGE_ACCOUNT_NAME", "lab5mri")
STORAGE_ACCOUNT_KEY = os.environ.get("STORAGE_ACCOUNT_KEY")  # set as env var
CONTAINER_NAME = "lakehouse"

# Path inside the container (Bronze layer)
BRONZE_PREFIX = "raw/tumor_images"  # will become raw/tumor_images/yes/..., /no/...


# --------- UPLOAD LOGIC (IDEMPOTENT) ---------

def upload_label_folder(container_client, label: str):
    """
    Upload all files from LOCAL_DATASET_DIR/<label> to
    lakehouse/raw/tumor_images/<label>/ in ADLS Gen2.

    Idempotent: skips files that already exist.
    """
    local_dir = LOCAL_DATASET_DIR / label
    if not local_dir.exists():
        raise FileNotFoundError(f"Local folder not found: {local_dir}")

    files = [p for p in local_dir.iterdir() if p.is_file()]
    print(f"Found {len(files)} files in {local_dir}")

    uploaded = 0
    skipped = 0

    for path in files:
        blob_path = f"{BRONZE_PREFIX}/{label}/{path.name}"
        blob_client = container_client.get_blob_client(blob_path)

        # Idempotency: do not re-upload if it already exists
        if blob_client.exists():
            print(f"[SKIP] {blob_path} already exists")
            skipped += 1
            continue

        with open(path, "rb") as f:
            blob_client.upload_blob(f)
        print(f"[UPLOAD] {blob_path}")
        uploaded += 1

    return uploaded, skipped


def main():
    if STORAGE_ACCOUNT_KEY is None:
        raise ValueError(
            "Please set the STORAGE_ACCOUNT_KEY environment variable before running this script."
        )

    account_url = f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net"
    service_client = BlobServiceClient(
        account_url=account_url, credential=STORAGE_ACCOUNT_KEY
    )
    container_client = service_client.get_container_client(CONTAINER_NAME)

    start_time = time()

    total_uploaded = 0
    total_skipped = 0

    for label in ["yes", "no"]:
        print(f"\n=== Uploading label: {label} ===")
        uploaded, skipped = upload_label_folder(container_client, label)
        total_uploaded += uploaded
        total_skipped += skipped

    elapsed = time() - start_time
    print("\n=== Upload summary ===")
    print(f"Uploaded: {total_uploaded} files")
    print(f"Skipped (already existed): {total_skipped} files")
    print(f"Total time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
