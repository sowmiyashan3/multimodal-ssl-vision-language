import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class ValDataset(Dataset):
    def __init__(self, csv_path, image_root, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_root, row["s3_key"])
        caption = row["caption"]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, caption
