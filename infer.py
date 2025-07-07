import argparse
import torch
from utils.misc import load_config
from datasets.prcv_dataset import PRCVDataset
from models import MultiModalReIDModel
from engine.eval import evaluate
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--ckpt', type=str, required=True, help='模型权重路径')
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device(cfg['train']['device'] if torch.cuda.is_available() else 'cpu')
    # 数据集
    query_dataset = PRCVDataset(cfg['data']['val_dir'], mode='val', modalities=cfg['data']['modalities'])
    gallery_dataset = PRCVDataset(cfg['data']['val_dir'], mode='val', modalities=['vis'])  # gallery只用RGB
    # 模型
    model = MultiModalReIDModel(cfg).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    # 推理
    evaluate(cfg, model, query_dataset, gallery_dataset, device)

if __name__ == '__main__':
    main() 