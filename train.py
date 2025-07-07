import argparse
import os
import torch
from utils.misc import load_config
from datasets.prcv_dataset import PRCVDataset
from models import MultiModalReIDModel
from engine.train import train
import torch.nn as nn
from utils.losses import TripletLoss
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device(cfg['train']['device'] if torch.cuda.is_available() else 'cpu')
    os.makedirs(cfg['output']['save_dir'], exist_ok=True)
    # 数据集
    train_dataset = PRCVDataset(cfg['data']['train_dir'], mode='train', modalities=cfg['data']['modalities'])
    # 模型
    model = MultiModalReIDModel(cfg).to(device)
    # 损失
    criterion_ce = nn.CrossEntropyLoss()
    criterion_tri = TripletLoss(margin=cfg['train'].get('triplet_margin', 0.3)) if cfg['train'].get('use_triplet', True) else None
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['train']['lr'])
    # 训练
    train(cfg, model, train_dataset, optimizer, criterion_ce, criterion_tri, device)

if __name__ == '__main__':
    main() 