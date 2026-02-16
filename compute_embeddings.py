import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import BertTokenizer
from tqdm import tqdm
import os

from dataset import ValDataset
from model import ImageTextModel

DEVICE = "cpu"
MODEL_PATH = "models/latest.pt"
VAL_CSV = "data/cc_1m_split_val.csv"
IMAGE_ROOT = "data"
SAVE_PATH = "val_embeddings.pt"

BATCH_SIZE = 128

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

dataset = ValDataset(VAL_CSV, IMAGE_ROOT, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model = ImageTextModel()
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model"])
model.to(DEVICE)
model.eval()

all_img_embeds = []
all_txt_embeds = []

with torch.no_grad():
    for images, captions in tqdm(dataloader):
        images = images.to(DEVICE)

        enc = tokenizer(
            list(captions),
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt"
        ).to(DEVICE)

        img_embeds, txt_embeds = model(
            images,
            enc.input_ids,
            enc.attention_mask
        )

        all_img_embeds.append(img_embeds.cpu())
        all_txt_embeds.append(txt_embeds.cpu())

# Concatenate
all_img_embeds = torch.cat(all_img_embeds)
all_txt_embeds = torch.cat(all_txt_embeds)

torch.save({
    "image_embeds": all_img_embeds,
    "text_embeds": all_txt_embeds
}, SAVE_PATH)

print(f"Saved embeddings to {SAVE_PATH}")
