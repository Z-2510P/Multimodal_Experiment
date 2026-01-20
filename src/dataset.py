import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import re

class MultiModalDataset(Dataset):
    def __init__(self, data_dir, label_file, transform=None, tokenizer=None, mode='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.mode = mode
        
        # 读取 CSV 格式的 txt 文件
        self.data_df = pd.read_csv(label_file)
        
        # 标签映射
        self.label_map = {'positive': 0, 'neutral': 1, 'negative': 2, 'null': -1}

    def clean_text(self, text):
        text = str(text)
        text = re.sub(r'http\S+', '', text) # 去链接
        text = re.sub(r'@\w+', '', text)    # 去用户提及
        return text.strip()

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        guid = str(row['guid'])
        
        # 1. 处理标签
        tag = row['tag'] if 'tag' in row else 'null'
        label = self.label_map.get(tag, -1)
        
        # 2. 处理文本
        txt_path = os.path.join(self.data_dir, f"{guid}.txt")
        text_content = ""
        try:
            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                text_content = f.read()
        except:
            text_content = "" # 容错
            
        text_content = self.clean_text(text_content)
        
        if self.tokenizer:
            encoding = self.tokenizer(
                text_content,
                padding='max_length',
                truncation=True,
                max_length=80, # 文本长度限制
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
        else:
            input_ids = torch.tensor([])
            attention_mask = torch.tensor([])

        # 3. 处理图片
        img_path = os.path.join(self.data_dir, f"{guid}.jpg")
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except:
            # 图片损坏时的兜底策略：全黑图片
            image = torch.zeros((3, 224, 224))

        return {
            'guid': guid,
            'image': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long)
        }