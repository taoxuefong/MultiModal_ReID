import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        # embeddings: [B, D], labels: [B]
        dist = torch.cdist(embeddings, embeddings)
        mask_pos = labels.unsqueeze(1) == labels.unsqueeze(0)
        mask_neg = labels.unsqueeze(1) != labels.unsqueeze(0)
        hardest_pos = (dist * mask_pos.float()).max(1)[0]
        hardest_neg = (dist + 1e5 * mask_pos.float()).min(1)[0]
        loss = F.relu(hardest_pos - hardest_neg + self.margin).mean()
        return loss

class MultiModalContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, img_emb, txt_emb):
        # img_emb, txt_emb: [B, D]
        logits = torch.mm(img_emb, txt_emb.t()) / self.temperature
        labels = torch.arange(img_emb.size(0)).to(img_emb.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        return (loss_i2t + loss_t2i) / 2 