import torch
import torch.nn as nn

class MultiModalFusion(nn.Module):
    def __init__(self, input_dims, fusion_type='attention'):
        super().__init__()
        self.fusion_type = fusion_type
        total_dim = sum(input_dims)
        self.num_modals = len(input_dims)
        if fusion_type == 'concat':
            self.fusion = nn.Linear(total_dim, 512)
        elif fusion_type == 'attention':
            self.attn = nn.Parameter(torch.ones(self.num_modals))
            self.fusion = nn.Linear(total_dim, 512)
        elif fusion_type == 'gated':
            self.gates = nn.ModuleList([nn.Sequential(
                nn.Linear(dim, 1), nn.Sigmoid()
            ) for dim in input_dims])
            self.fusion = nn.Linear(total_dim, 512)
        else:
            raise NotImplementedError

    def forward(self, features, mask=None):
        # features: list of [B, D]，mask: [B, num_modals]
        if self.fusion_type == 'attention' and mask is not None:
            attn_weights = torch.softmax(self.attn, dim=0)  # [num_modals]
            # batch级mask加权
            attn_weights = attn_weights.unsqueeze(0) * mask  # [B, num_modals]
            attn_weights = attn_weights / (attn_weights.sum(dim=1, keepdim=True) + 1e-6)
            features = [f * attn_weights[:, i:i+1] for i, f in enumerate(features)]
        elif self.fusion_type == 'gated':
            gated_feats = []
            for i, f in enumerate(features):
                gate = self.gates[i](f)
                if mask is not None:
                    gate = gate * mask[:, i:i+1]
                gated_feats.append(f * gate)
            features = gated_feats
        x = torch.cat(features, dim=1)
        return self.fusion(x) 