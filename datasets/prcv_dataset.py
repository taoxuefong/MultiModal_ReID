import os
import json
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import BertTokenizer
import torchvision.transforms as T

class PRCVDataset(Dataset):
    def __init__(self, root, mode='train', modalities=['vis', 'nir', 'cp', 'sk', 'text'], transform=None):
        self.root = root
        self.mode = mode
        self.modalities = modalities
        self.transform = transform
        self.text_annos = self._load_text_annos()
        self.data = self._load_data()
        self.tokenizer = BertTokenizer.from_pretrained('/data/taoxuefeng/PRCV/bert-base-uncased')
        self.img_shape = (3, 224, 224)
        self.text_len = 64
        self.default_img_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])

    def _load_text_annos(self):
        text_json = os.path.join(self.root, 'text_annos.json')
        if not os.path.exists(text_json):
            return {}
        with open(text_json, 'r') as f:
            text_list = json.load(f)
        text_dict = {}
        for item in text_list:
            if item['split'] == self.mode:
                text_dict[item['file_path']] = {
                    'caption': item['caption'],
                    'id': item['id']
                }
        return text_dict

    def _load_data(self):
        data = []
        vis_dir = os.path.join(self.root, 'vis')
        for id_folder in os.listdir(vis_dir):
            id_path = os.path.join(vis_dir, id_folder)
            if not os.path.isdir(id_path):
                continue
            for fname in os.listdir(id_path):
                vis_path = os.path.join('vis', id_folder, fname)
                item = {'vis': os.path.join(self.root, vis_path)}
                item['id'] = int(id_folder)
                base_name = fname.split('_vis.')[0]
                # 红外
                nir_name = base_name + '_nir.jpg'
                nir_path = os.path.join(self.root, 'nir', id_folder, nir_name)
                item['nir'] = nir_path if os.path.exists(nir_path) else None
                # 彩铅
                cp_name = base_name + '_colorpencil.jpg'
                cp_path = os.path.join(self.root, 'cp', id_folder, cp_name)
                item['cp'] = cp_path if os.path.exists(cp_path) else None
                # 素描
                sk_name = base_name + '_sketch.jpg'
                sk_path = os.path.join(self.root, 'sk', id_folder, sk_name)
                item['sk'] = sk_path if os.path.exists(sk_path) else None
                # 文本
                if vis_path in self.text_annos:
                    item['text'] = self.text_annos[vis_path]['caption']
                else:
                    item['text'] = None
                data.append(item)
        return data

    def __getitem__(self, idx):
        item = self.data[idx]
        sample = {}
        mask = {}
        for m in self.modalities:
            if m == 'text':
                sample[m] = item[m]
                mask[m] = 1 if item[m] is not None else 0
            else:
                if item[m] is not None:
                    img = Image.open(item[m]).convert('RGB')
                    if self.transform:
                        img = self.transform(img)
                    else:
                        img = self.default_img_transform(img)
                    sample[m] = img
                    mask[m] = 1
                else:
                    sample[m] = torch.zeros(self.img_shape, dtype=torch.float32)
                    mask[m] = 0
        # 随机mask部分模态（训练时可选）
        if self.mode == 'train':
            for m in self.modalities:
                if random.random() < 0.1:
                    if m == 'text':
                        sample[m] = None
                    else:
                        sample[m] = torch.zeros(self.img_shape, dtype=torch.float32)
                    mask[m] = 0
        # 文本转input_ids/attention_mask
        if sample.get('text', None):
            encoding = self.tokenizer(
                sample['text'], truncation=True, padding='max_length', max_length=self.text_len, return_tensors='pt'
            )
            sample['text_input_ids'] = encoding['input_ids'].squeeze(0)
            sample['text_attention_mask'] = encoding['attention_mask'].squeeze(0)
        else:
            sample['text_input_ids'] = torch.zeros(self.text_len, dtype=torch.long)
            sample['text_attention_mask'] = torch.zeros(self.text_len, dtype=torch.long)
        sample['id'] = item['id']
        sample['mask'] = mask
        return sample

    def __len__(self):
        return len(self.data) 