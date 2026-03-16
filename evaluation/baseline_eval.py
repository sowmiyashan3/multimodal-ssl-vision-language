# baseline_eval.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from dataset import ValDataset
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VAL_CSV = "data/cc_1m_split_val.csv"
IMAGE_ROOT = "data"
BATCH_SIZE = 128

# ---------------- Transform ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# ---------------- Dataset & Dataloader ----------------
dataset = ValDataset(VAL_CSV, IMAGE_ROOT, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------- Models ----------------
# Pretrained ViT
vit = models.vit_b_16(weights='IMAGENET1K_V1')
vit.heads = torch.nn.Identity()  # remove classifier
vit.to(DEVICE)
vit.eval()

# Pretrained BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased").to(DEVICE)
bert.eval()

# ---------------- Compute embeddings ----------------
all_img_embeds = []
all_txt_embeds = []

with torch.no_grad():
    for images, captions in tqdm(dataloader, desc="Computing embeddings"):
        images = images.to(DEVICE)
        img_embeds = vit(images)
        img_embeds = F.normalize(img_embeds, dim=-1)
        all_img_embeds.append(img_embeds.cpu())

        enc = tokenizer(list(captions), padding=True, truncation=True,
                        max_length=64, return_tensors="pt").to(DEVICE)
        txt_embeds = bert(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
        txt_embeds = F.normalize(txt_embeds.last_hidden_state[:,0,:], dim=-1)  # CLS token
        all_txt_embeds.append(txt_embeds.cpu())

all_img_embeds = torch.cat(all_img_embeds)
all_txt_embeds = torch.cat(all_txt_embeds)

# ---------------- Compute Recall@K ----------------
# Compute cosine similarity
sim_matrix = all_txt_embeds @ all_img_embeds.t()

top1 = torch.topk(sim_matrix, k=1, dim=1).indices
top5 = torch.topk(sim_matrix, k=5, dim=1).indices
top10 = torch.topk(sim_matrix, k=10, dim=1).indices

labels = torch.arange(sim_matrix.size(0))

recall1 = (top1.squeeze() == labels).float().mean().item()
recall5 = (top5 == labels.unsqueeze(1)).any(dim=1).float().mean().item()
recall10 = (top10 == labels.unsqueeze(1)).any(dim=1).float().mean().item()

print(f"ViT+BERT Baseline:")
print(f"Recall@1: {recall1:.4f}")
print(f"Recall@5: {recall5:.4f}")
print(f"Recall@10: {recall10:.4f}")
