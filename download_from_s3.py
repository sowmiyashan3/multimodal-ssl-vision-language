import os
import boto3
from tqdm import tqdm

# ---------------- CONFIG ----------------
BUCKET = "sowmi-ssl-project-images"

CSV_S3_KEY = "resized_224/cc_1m_split_train.csv"
LOCAL_CSV = "/data/ssl-project/data/cc_1m_split_train.csv"

IMAGE_PREFIX = "train/resized_224/"
LOCAL_IMAGE_ROOT = "/data/ssl-project/data/images/"

os.makedirs(os.path.dirname(LOCAL_CSV), exist_ok=True)
os.makedirs(LOCAL_IMAGE_ROOT, exist_ok=True)

# ---------------- S3 CLIENT ----------------
s3 = boto3.client("s3")

# ---------------- DOWNLOAD CSV ----------------
if not os.path.exists(LOCAL_CSV):
    print(f"Downloading CSV: {CSV_S3_KEY} ...")
    s3.download_file(BUCKET, CSV_S3_KEY, LOCAL_CSV)
    print(f"CSV downloaded to {LOCAL_CSV}")
else:
    print(f"CSV already exists at {LOCAL_CSV}")

# ---------------- DOWNLOAD IMAGES ----------------
print("Downloading images from S3 ...")
paginator = s3.get_paginator("list_objects_v2")
pages = paginator.paginate(Bucket=BUCKET, Prefix=IMAGE_PREFIX)

total_size = 0
num_files = 0

for page in pages:
    for obj in tqdm(page.get("Contents", []), desc="Images"):
        key = obj["Key"]
        if key.endswith("/"):  # skip folders
            continue

        local_path = os.path.join(LOCAL_IMAGE_ROOT, key)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        if not os.path.exists(local_path):
            s3.download_file(BUCKET, key, local_path)
            size_mb = os.path.getsize(local_path) / (1024 * 1024)
            total_size += size_mb
            num_files += 1

print(f"\nDownloaded {num_files} new images")
print(f"Total disk space used: {total_size:.2f} MB")
print("All files are on local disk. No images are loaded into RAM.")
