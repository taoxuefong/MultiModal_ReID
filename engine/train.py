import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from datasets.sampler import RandomIdentitySampler
from utils.losses import TripletLoss, MultiModalContrastiveLoss


def train(cfg, model, dataset, optimizer, criterion_ce, criterion_tri, device):
    # 采样器
    sampler = RandomIdentitySampler(dataset, num_instances=cfg['train'].get('num_instances', 4))
    loader = DataLoader(dataset, batch_size=cfg['train']['batch_size'], sampler=sampler, num_workers=cfg['train'].get('num_workers', 4), drop_last=True)
    model.train()
    contrastive_loss = MultiModalContrastiveLoss() if cfg['train'].get('use_contrastive', True) else None
    for epoch in range(cfg['train']['epochs']):
        total_loss = 0
        for batch in tqdm(loader, desc=f'Epoch {epoch+1}/{cfg["train"]["epochs"]}'):
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            mask = batch['mask']
            out = model(batch, mask)
            # 分类损失
            loss_ce = criterion_ce(out, batch['id'])
            # Triplet损失
            loss_tri = 0
            if criterion_tri is not None:
                loss_tri = criterion_tri(out, batch['id'])
            # 文本-图像对齐损失
            loss_contrast = 0
            if contrastive_loss is not None and batch['text_input_ids'] is not None:
                # 只对有文本的样本做对齐
                img_emb = out  # 假设out为融合embedding
                txt_emb = model.text_encoder(batch['text_input_ids'], batch['text_attention_mask'])
                loss_contrast = contrastive_loss(img_emb, txt_emb)
            loss = loss_ce + loss_tri + loss_contrast
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}')
        torch.save(model.state_dict(), f'{cfg["output"]["save_dir"]}/model_epoch{epoch+1}.pth') 