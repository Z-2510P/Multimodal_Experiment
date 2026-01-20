import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from transformers import BertModel

class MultimodalModel(nn.Module):
    def __init__(self, num_classes=3):
        super(MultimodalModel, self).__init__()
        
        # Image Encoder
        weights = ResNet50_Weights.DEFAULT
        self.resnet = resnet50(weights=weights)
        self.resnet.fc = nn.Identity() # 移除全连接层
        self.img_hidden_size = 2048

        # Text Encoder
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.text_hidden_size = 768

        # Fusion Layer
        fusion_dim = self.img_hidden_size + self.text_hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, input_ids, attention_mask, image):
        # 提取文本特征 ([CLS] token)
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = bert_output.pooler_output 

        # 提取图像特征
        img_features = self.resnet(image)
        
        # 拼接
        combined_features = torch.cat((text_features, img_features), dim=1)
        
        # 分类
        logits = self.classifier(combined_features)
        return logits