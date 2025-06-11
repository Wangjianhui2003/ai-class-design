"""
基于Transformer的无人机图像特征提取
使用Vision Transformer (ViT)对VisDrone数据集进行目标区域特征提取和可视化
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import ViTFeatureExtractor, ViTModel
import cv2
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class VisDroneDataset:
    """VisDrone数据集处理类"""
    
    def __init__(self, image_dir, annotation_dir):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.class_names = {
            0: '忽略',
            1: '行人',
            2: '人',
            3: '自行车',
            4: '汽车',
            5: '货车',
            6: '三轮车',
            7: '遮阳篷-三轮车',
            8: '公交车',
            9: '机动车',
            10: '其他'
        }
    
    def load_annotations(self, annotation_file):
        """加载标注文件"""
        annotations = []
        with open(annotation_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(',')
                    if len(parts) >= 8:
                        x, y, w, h = map(int, parts[:4])
                        score = int(parts[4])
                        category = int(parts[5])
                        truncation = int(parts[6])
                        occlusion = int(parts[7])
                        
                        # 过滤掉忽略的对象和太小的对象
                        if category > 0 and w > 20 and h > 20:
                            annotations.append({
                                'bbox': [x, y, w, h],
                                'category': category,
                                'score': score,
                                'truncation': truncation,
                                'occlusion': occlusion
                            })
        return annotations
    
    def get_sample_data(self, num_samples=10):
        """获取样本数据"""
        image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.jpg')]
        image_files = sorted(image_files)[:num_samples]
        
        samples = []
        for img_file in image_files:
            # 构建对应的标注文件名
            base_name = os.path.splitext(img_file)[0]
            ann_file = base_name + '.txt'
            
            img_path = os.path.join(self.image_dir, img_file)
            ann_path = os.path.join(self.annotation_dir, ann_file)
            
            if os.path.exists(ann_path):
                annotations = self.load_annotations(ann_path)
                if annotations:  # 只保留有有效标注的图像
                    samples.append({
                        'image_path': img_path,
                        'annotations': annotations,
                        'filename': img_file
                    })
        
        return samples

class ViTFeatureExtractor_Custom:
    """基于Vision Transformer的特征提取器"""
    
    def __init__(self, model_name='google/vit-base-patch16-224'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载预训练的ViT模型
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, image, bbox=None):
        """提取图像特征"""
        if bbox is not None:
            # 如果提供了边界框，则裁剪目标区域
            x, y, w, h = bbox
            image = image.crop((x, y, x + w, y + h))
        
        # 预处理图像
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 提取特征
            outputs = self.model(image_tensor)
            # 使用CLS token的特征作为全局特征
            features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
        return features.flatten()

def visualize_detection_results(samples, feature_extractor, dataset, max_samples=5):
    """可视化检测结果和特征提取"""
    fig, axes = plt.subplots(2, max_samples, figsize=(20, 8))
    if max_samples == 1:
        axes = axes.reshape(2, 1)
    
    all_features = []
    all_labels = []
    
    for idx, sample in enumerate(samples[:max_samples]):
        # 加载图像
        image = Image.open(sample['image_path']).convert('RGB')
        image_np = np.array(image)
        
        # 显示原图和检测框
        ax1 = axes[0, idx]
        ax1.imshow(image_np)
        ax1.set_title(f"原图: {sample['filename']}")
        ax1.axis('off')
        
        # 绘制边界框
        for ann in sample['annotations'][:5]:  # 只显示前5个目标
            bbox = ann['bbox']
            category = ann['category']
            
            rect = patches.Rectangle(
                (bbox[0], bbox[1]), bbox[2], bbox[3],
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax1.add_patch(rect)
            
            # 添加类别标签
            class_name = dataset.class_names.get(category, '未知')
            ax1.text(bbox[0], bbox[1]-5, f'{class_name}', 
                    color='red', fontsize=8, weight='bold')
        
        # 提取目标区域特征并可视化
        target_crops = []
        target_features = []
        
        for ann in sample['annotations'][:3]:  # 处理前3个目标
            bbox = ann['bbox']
            category = ann['category']
            
            # 提取目标区域特征
            features = feature_extractor.extract_features(image, bbox)
            target_features.append(features)
            target_crops.append(image.crop((bbox[0], bbox[1], 
                                          bbox[0] + bbox[2], bbox[1] + bbox[3])))
            
            all_features.append(features)
            all_labels.append(category)
        
        # 显示第一个目标区域
        if target_crops:
            ax2 = axes[1, idx]
            ax2.imshow(target_crops[0])
            first_label = all_labels[-len(target_crops)]
            ax2.set_title(f"目标区域 (类别: {dataset.class_names.get(first_label, '未知')})")
            ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('detection_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return np.array(all_features), np.array(all_labels)

def visualize_feature_analysis(features, labels, dataset):
    """可视化特征分析"""
    # 使用PCA降维
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)
    
    # 使用t-SNE降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
    features_tsne = tsne.fit_transform(features)
    
    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # PCA可视化
    ax1 = axes[0, 0]
    unique_labels = np.unique(labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax1.scatter(features_pca[mask, 0], features_pca[mask, 1], 
                   c=[colors[i]], label=dataset.class_names.get(label, f'类别{label}'),
                   alpha=0.7, s=60)
    
    ax1.set_xlabel(f'第一主成分 (解释方差: {pca.explained_variance_ratio_[0]:.2%})')
    ax1.set_ylabel(f'第二主成分 (解释方差: {pca.explained_variance_ratio_[1]:.2%})')
    ax1.set_title('PCA特征可视化')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # t-SNE可视化
    ax2 = axes[0, 1]
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax2.scatter(features_tsne[mask, 0], features_tsne[mask, 1], 
                   c=[colors[i]], label=dataset.class_names.get(label, f'类别{label}'),
                   alpha=0.7, s=60)
    
    ax2.set_xlabel('t-SNE 维度1')
    ax2.set_ylabel('t-SNE 维度2')
    ax2.set_title('t-SNE特征可视化')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 特征维度分布
    ax3 = axes[1, 0]
    feature_means = np.mean(features, axis=0)
    ax3.plot(feature_means)
    ax3.set_xlabel('特征维度')
    ax3.set_ylabel('平均特征值')
    ax3.set_title('特征向量分布')
    ax3.grid(True, alpha=0.3)
    
    # 类别统计
    ax4 = axes[1, 1]
    unique_labels, counts = np.unique(labels, return_counts=True)
    class_names = [dataset.class_names.get(label, f'类别{label}') for label in unique_labels]
    
    bars = ax4.bar(range(len(unique_labels)), counts, color=colors[:len(unique_labels)])
    ax4.set_xlabel('目标类别')
    ax4.set_ylabel('数量')
    ax4.set_title('目标类别分布')
    ax4.set_xticks(range(len(unique_labels)))
    ax4.set_xticklabels(class_names, rotation=45, ha='right')
    
    # 在柱状图上添加数值
    for bar, count in zip(bars, counts):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('feature_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_feature_similarity(features, labels, dataset, top_k=5):
    """分析特征相似性"""
    print("\n=== 特征相似性分析 ===")
    
    # 计算特征相似性矩阵
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(features)
    
    # 创建相似性热图
    plt.figure(figsize=(10, 8))
    
    # 创建标签
    label_names = [dataset.class_names.get(label, f'类别{label}') for label in labels]
    
    # 绘制热图
    sns.heatmap(similarity_matrix, 
                xticklabels=label_names, 
                yticklabels=label_names,
                cmap='viridis', 
                center=0,
                square=True,
                fmt='.2f')
    
    plt.title('特征余弦相似性矩阵')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('similarity_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 分析每个类别的平均特征
    unique_labels = np.unique(labels)
    class_features = {}
    
    for label in unique_labels:
        mask = labels == label
        class_features[label] = np.mean(features[mask], axis=0)
    
    # 计算类别间相似性
    print("\n类别间特征相似性:")
    for i, label1 in enumerate(unique_labels):
        for j, label2 in enumerate(unique_labels):
            if i < j:
                sim = cosine_similarity([class_features[label1]], [class_features[label2]])[0][0]
                class1_name = dataset.class_names.get(label1, f'类别{label1}')
                class2_name = dataset.class_names.get(label2, f'类别{label2}')
                print(f"{class1_name} vs {class2_name}: {sim:.3f}")

def main():
    """主函数"""
    print("=== 基于Transformer的无人机图像特征提取 ===")
    print("使用VisDrone数据集和Vision Transformer模型\n")
    
    # 数据路径
    image_dir = "dataset/images"
    annotation_dir = "dataset/annotations"
    
    # 检查数据路径
    if not os.path.exists(image_dir) or not os.path.exists(annotation_dir):
        print("错误: 数据集路径不存在!")
        print(f"图像路径: {image_dir}")
        print(f"标注路径: {annotation_dir}")
        return
    
    # 初始化数据集
    dataset = VisDroneDataset(image_dir, annotation_dir)
    print("数据集初始化完成")
    
    # 获取样本数据
    samples = dataset.get_sample_data(num_samples=8)
    print(f"加载了 {len(samples)} 个样本")
    
    if len(samples) == 0:
        print("错误: 没有找到有效的样本数据!")
        return
    
    # 初始化特征提取器
    print("\n正在加载Vision Transformer模型...")
    feature_extractor = ViTFeatureExtractor_Custom()
    
    print("模型加载完成，开始特征提取")
    
    # 可视化检测结果和提取特征
    features, labels = visualize_detection_results(samples, feature_extractor, dataset, max_samples=5)
    
    print(f"\n提取了 {len(features)} 个目标的特征")
    print(f"特征维度: {features.shape[1]}")
    print(f"涉及类别数: {len(np.unique(labels))}")
    
    # 特征分析和可视化
    if len(features) > 1:
        print("\n开始特征分析...")
        visualize_feature_analysis(features, labels, dataset)
        
        # 相似性分析
        if len(features) > 2:
            analyze_feature_similarity(features, labels, dataset)
    
    # 输出统计信息
    print("\n=== 统计信息 ===")
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        class_name = dataset.class_names.get(label, f'类别{label}')
        print(f"{class_name}: {count} 个目标")
    
    print(f"\n特征向量统计:")
    print(f"平均值: {np.mean(features):.4f}")
    print(f"标准差: {np.std(features):.4f}")
    print(f"最小值: {np.min(features):.4f}")
    print(f"最大值: {np.max(features):.4f}")
    
    print("\n=== 分析完成 ===")
    print("生成的文件:")
    print("- detection_results.png: 检测结果可视化")
    print("- feature_analysis.png: 特征分析可视化")
    print("- similarity_matrix.png: 特征相似性矩阵")

if __name__ == "__main__":
    main()