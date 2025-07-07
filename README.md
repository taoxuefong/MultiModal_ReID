# MultiModal_ReID

本项目支持RGB、红外、彩铅、素描、文本五种模态，支持模态缺失和多模态组合检索，参考MMANet的模态缺失处理方法，并新增文本模态支持。

## 目录结构

- data/: 数据集目录（需自行放置）
- models/: 模型主干、各模态头、融合模块
- datasets/: 数据加载与处理
- engine/: 训练与推理主程序
- utils/: 工具函数与评测指标
- configs/: 配置文件
- train.py: 训练入口
- infer.py: 推理/生成csv入口

## 快速开始

1. 安装依赖  
   `pip install -r requirements.txt`

2. 配置数据集路径与参数（见configs/config.yaml）

3. 训练  
   `python train.py --config configs/config.yaml`

4. 推理/生成csv  
   `python infer.py --config configs/config.yaml`

## 主要特性

- 支持任意模态组合检索
- 支持文本模态（BERT/CLIP编码）
- 代码结构清晰，便于扩展 
