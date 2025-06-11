"""
基于Transformer的无人机图像特征提取 - 演示代码
简化版本，用于展示核心功能
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from transformers import ViTFeatureExtractor, ViTModel
import torchvision.transforms as transforms

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def demo_feature_extraction():
    """演示特征提取功能"""
    print("=== ViT特征提取演示 ===")
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载预训练的ViT模型
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    model = ViTModel.from_pretrained('google/vit-base-patch16-224')
    model.to(device)
    model.eval()
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 加载一张示例图像
    image_path = "dataset/images/0000001_02999_d_0000005.jpg"

    if not os.path.exists(image_path):
        print(f"找不到示例图像: {image_path}")
        return

    image = Image.open(image_path).convert('RGB')
    print(f"图像尺寸: {image.size}")
    
    # 预处理
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 提取特征
    with torch.no_grad():
        outputs = model(image_tensor)
        # 使用CLS token作为全局特征
        features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
    print(f"提取的特征维度: {features.shape}")
    print(f"特征向量前10个值: {features[0][:10]}")
    
    # 简单可视化
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("原始无人机图像")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.plot(features[0])
    plt.title("ViT特征向量")
    plt.xlabel("特征维度")
    plt.ylabel("特征值")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('demo_feature_extraction.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("生成demo_feature_extraction.png")
        

if __name__ == "__main__":
    demo_feature_extraction()
