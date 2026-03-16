import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class SSLImageCaptionDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
	Args:
            csv_file (str): Path to CSV with columns: image_id, s3_key, caption, local_path
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['local_path']
        caption = row['caption']

        # Load image from disk
        image = Image.open(img_path).convert("RGB")

        # Apply transforms if any
        if self.transform:
            image = self.transform(image)

        return image, caption

# ---------------- Example Usage ----------------
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import torch

    CSV_PATH = "/data/ssl-project/data/cc_1m_split_train.csv"

    # Example transforms (ViT/SSL-style)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = SSLImageCaptionDataset(CSV_PATH, transform=transform)

    # Use multiple workers for fast disk loading
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True,
                            num_workers=8, pin_memory=True)

    # Test one batch
    for images, captions in dataloader:
        print(images.shape)   # torch.Size([64, 3, 224, 224])
        print(captions[:5])  # first 5 captions
        break

