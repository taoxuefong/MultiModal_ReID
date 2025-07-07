import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

def evaluate(cfg, model, query_dataset, gallery_dataset, device):
    query_loader = DataLoader(query_dataset, batch_size=cfg['train']['batch_size'], shuffle=False, num_workers=cfg['train'].get('num_workers', 4))
    gallery_loader = DataLoader(gallery_dataset, batch_size=cfg['train']['batch_size'], shuffle=False, num_workers=cfg['train'].get('num_workers', 4))
    model.eval()
    query_features, query_types, query_indices = [], [], []
    gallery_features, gallery_indices = [], []
    with torch.no_grad():
        for batch in tqdm(query_loader, desc='Extracting query features'):
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            mask = batch['mask']
            feat = model(batch, mask)
            query_features.append(feat.cpu())
            query_types.extend([get_query_type(batch, cfg['data']['modalities'])] * feat.size(0))
            query_indices.extend(batch['id'].cpu().tolist())
        for batch in tqdm(gallery_loader, desc='Extracting gallery features'):
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            mask = batch['mask']
            feat = model(batch, mask)
            gallery_features.append(feat.cpu())
            gallery_indices.extend(batch['id'].cpu().tolist())
    query_features = torch.cat(query_features, dim=0)
    gallery_features = torch.cat(gallery_features, dim=0)
    # 计算相似度
    sim_matrix = torch.mm(query_features, gallery_features.t())
    ranking = torch.argsort(sim_matrix, dim=1, descending=True)
    results = []
    for i, idxs in enumerate(ranking):
        results.append({
            'query_idx': query_indices[i],
            'query_type': query_types[i],
            'ranking_list_idx': [gallery_indices[j] for j in idxs.cpu().numpy().tolist()]
        })
    df = pd.DataFrame(results)
    df.to_csv(cfg['output']['csv_path'], index=False)
    print(f'Result saved to {cfg["output"]["csv_path"]}')

def get_query_type(batch, modalities):
    # 根据batch中实际存在的模态生成query_type字符串
    present = [m.upper() for m in modalities if batch['mask'][m][0] == 1]
    if len(present) == 1:
        return f'onemodal_{present[0]}'
    else:
        return f'{len(present)}modal_' + '_'.join(present) 