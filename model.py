# model.py
import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertModel, BertTokenizer

class ImageTextModel(nn.Module):
    def __init__(self, image_embed_dim=768, text_embed_dim=768):
        super().__init__()

        # -------- Image Encoder (ViT) --------
        self.vit = models.vit_b_16(weights='IMAGENET1K_V1')
        self.vit.heads = nn.Identity()  # remove classifier
        self.image_proj = nn.Linear(768, image_embed_dim)

        # -------- Text Encoder (BERT) --------
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.text_proj = nn.Linear(text_embed_dim, image_embed_dim)

        # Temperature parameter for contrastive loss
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1/0.07)))

    def forward(self, images, input_ids, attention_mask):
        # Image embeddings
        img_embeds = self.vit(images)  # [B, 768]
        img_embeds = self.image_proj(img_embeds)
        img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)

        # Text embeddings
        txt_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        txt_embeds = txt_out.last_hidden_state[:, 0, :]  # [CLS] token
        txt_embeds = self.text_proj(txt_embeds)
        txt_embeds = txt_embeds / txt_embeds.norm(dim=-1, keepdim=True)

        return img_embeds, txt_embeds
