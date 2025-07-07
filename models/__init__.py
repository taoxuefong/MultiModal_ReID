import torch
import torch.nn as nn
from .backbone import VisualBackbone
from .modal_heads import TextEncoder
from .fusion import MultiModalFusion

class MultiModalReIDModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 初始化视觉模态编码器（共用主干）
        self.visual_backbone = VisualBackbone(cfg['model']['backbone'])
        # 文本编码器
        self.text_encoder = TextEncoder()
        # 融合层
        input_dims = [self.visual_backbone.out_dim]*4 + [self.text_encoder.out_dim]
        self.fusion = MultiModalFusion(input_dims, fusion_type=cfg['model']['fusion'])
        self.modalities = cfg['data']['modalities']

    def forward(self, batch, mask=None):
        features = []
        for i, m in enumerate(self.modalities):
            if m == 'text':
                if batch[m] is not None:
                    # 假设batch['text']已转为input_ids, attention_mask
                    text_feat = self.text_encoder(batch['text_input_ids'], batch['text_attention_mask'])
                    features.append(text_feat)
                else:
                    features.append(torch.zeros(batch['vis'].shape[0], self.text_encoder.out_dim, device=batch['vis'].device))
            else:
                if batch[m] is not None:
                    feat = self.visual_backbone(batch[m])
                    features.append(feat)
                else:
                    features.append(torch.zeros(batch['vis'].shape[0], self.visual_backbone.out_dim, device=batch['vis'].device))
        if mask is not None:
            mask_tensor = torch.stack([mask[m] for m in self.modalities], dim=1).float().to(features[0].device)
        else:
            mask_tensor = torch.ones((features[0].shape[0], len(self.modalities)), device=features[0].device)
        out = self.fusion(features, mask_tensor)
        return out 