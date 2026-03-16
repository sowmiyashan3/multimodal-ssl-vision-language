import os
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from transformers import BertTokenizer
from dataset import SSLImageCaptionDataset
from model import ImageTextModel
from tqdm import tqdm
import subprocess

S3_MODEL_DIR = "s3://sowmi-ssl-project-images/models/"

def upload_to_s3(local_path):
    subprocess.run(
        ["aws", "s3", "cp", local_path, S3_MODEL_DIR],
        check=True
    )


# ---------------- CONFIG ----------------
CSV_PATH = "/data/ssl-project/data/cc_1m_split_train.csv"
IMAGE_ROOT = "/data/ssl-project/data/images"
MODEL_DIR = "/data/ssl-project/models"
LOG_FILE = "/data/ssl-project/logs/train.log"

BATCH_SIZE = 64
NUM_EPOCHS = 10
LR = 1e-4
NUM_WORKERS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# ---------------- TOKENIZER ----------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# ---------------- DATASET ----------------
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

dataset = SSLImageCaptionDataset(CSV_PATH, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=NUM_WORKERS, pin_memory=True)

# ---------------- MODEL ----------------
model = ImageTextModel().to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR)

# Contrastive loss (InfoNCE)
def contrastive_loss(image_embeds, text_embeds, logit_scale):
    logits = logit_scale.exp() * image_embeds @ text_embeds.t()
    labels = torch.arange(len(logits)).to(logits.device)
    loss_i = nn.CrossEntropyLoss()(logits, labels)
    loss_t = nn.CrossEntropyLoss()(logits.t(), labels)
    return (loss_i + loss_t) / 2


# ---------------- RESUME ----------------
start_epoch = 0
latest_ckpt = os.path.join(MODEL_DIR, "latest.pt")
if os.path.exists(latest_ckpt):
    print("Loading checkpoint...")
    ckpt = torch.load(latest_ckpt)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch = ckpt["epoch"] + 1
    print(f"Resuming from epoch {start_epoch}")

# ---------------- TRAIN ----------------
for epoch in range(start_epoch, NUM_EPOCHS):
    model.train()
    total_loss = 0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}")

    for i, (images, captions) in pbar:
        images = images.to(DEVICE)

        # Tokenize captions
        enc = tokenizer(list(captions), padding=True, truncation=True,
                        max_length=64, return_tensors="pt").to(DEVICE)

        optimizer.zero_grad()
        img_embeds, txt_embeds = model(images, enc.input_ids, enc.attention_mask)
        loss = contrastive_loss(img_embeds, txt_embeds, model.logit_scale)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (i+1) % 1000 == 0:
            with open(LOG_FILE, "a") as f:
                f.write(f"Epoch {epoch}, Step {i+1}, Loss: {total_loss/(i+1):.4f}\n")

    # Save checkpoint after each epoch
    ckpt_path = os.path.join(MODEL_DIR, f"model_epoch_{epoch}.pt")
    torch.save({"model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch}, ckpt_path)
    torch.save({"model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch}, latest_ckpt)

    # Upload both to S3
    upload_to_s3(ckpt_path)
    upload_to_s3(latest_ckpt)

    print(f"Epoch {epoch} finished. Checkpoint saved at {ckpt_path}")

