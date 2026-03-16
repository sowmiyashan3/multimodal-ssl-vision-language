# clean_val_csv.py
import pandas as pd
import os

VAL_CSV = "data/cc_1m_split_val.csv"      # original CSV
IMAGE_ROOT = "data/val/resized_224"       # where images are downloaded
OUTPUT_CSV = "data/cc_1m_split_val_clean.csv"

# Load CSV
df = pd.read_csv(VAL_CSV)

# Build full local paths
def image_exists(row):
    img_path = os.path.join(IMAGE_ROOT, *row['s3_key'].split('/')[2:])  # remove "val/resized_224" prefix
    return os.path.exists(img_path)

# Check existence
df['exists'] = df.apply(image_exists, axis=1)

# Filter rows
df_clean = df[df['exists']].drop(columns=['exists']).reset_index(drop=True)

# Save clean CSV
df_clean.to_csv(OUTPUT_CSV, index=False)
print(f"Cleaned CSV saved to {OUTPUT_CSV}")
print(f"{len(df_clean)} images available out of {len(df)}")
