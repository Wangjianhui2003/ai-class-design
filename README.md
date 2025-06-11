# 基于Transformer的无人机图像特征提取

| 课设成员 | 贡献 |
|------|----|
| 王建辉  | A  |
|李劲陶|A|

## 项目概述

- **目标**: 使用ViT对无人机图像中的目标区域进行全局特征提取
- **数据集**: VisDrone无人机图像数据集
- **模型**: Google预训练的Vision Transformer (ViT-Base-Patch16-224)
- **功能**: 目标检测、特征提取、特征可视化、相似性分析

## 项目结构

```
PythonProject1/
├── main2.py              # 主程序：完整的特征提取和分析
├── demo.py               # 演示程序：简化的特征提取示例
├── README.md            # 项目设计文档
├── dataset/             # 数据集目录
│   ├── images/          # 无人机图像
│   └── annotations/     # 标注文件
└── 生成的结果文件:
    ├── detection_results.png    # 检测结果可视化
    ├── feature_analysis.png     # 特征分析可视化
    └── similarity_matrix.png    # 特征相似性矩阵
```

## 数据集格式

### VisDrone标注格式
VisDrone数据集地址：https://github.com/VisDrone/VisDrone-Dataset
每个标注文件包含多行，每行格式为：
```
x,y,w,h,score,object_category,truncation,occlusion
```
- `x,y`: 边界框左上角坐标
- `w,h`: 边界框宽度和高度
- `score`: 置信度分数
- `object_category`: 目标类别（1-10）
- `truncation`: 截断标记
- `occlusion`: 遮挡标记

### 目标类别
- 0: 忽略
- 1: 行人
- 2: 人
- 3: 自行车
- 4: 汽车
- 5: 货车
- 6: 三轮车
- 7: 遮阳篷-三轮车
- 8: 公交车
- 9: 机动车
- 10: 其他

## 核心功能

### 1. 数据加载和预处理
- `VisDroneDataset` 类负责加载图像和标注
- 过滤无效目标（尺寸过小、类别为0）
- 支持批量处理多个样本

### 2. Vision Transformer特征提取
- `ViTFeatureExtractor_Custom` 类封装特征提取功能
- 使用预训练的ViT-Base模型
- 提取CLS token作为全局特征（768维）
- 支持全图和目标区域特征提取

### 3. 可视化和分析
- **检测结果可视化**: 显示原图、边界框和目标区域
- **特征分析**: PCA降维、t-SNE可视化、特征分布
- **相似性分析**: 计算特征余弦相似性矩阵
- **统计分析**: 类别分布、特征统计信息

## 技术特点

### 1. 模型架构
- **Vision Transformer**: 使用注意力机制处理图像patch
- **预训练权重**: 基于ImageNet预训练的google/vit-base-patch16-224
- **特征维度**: 768维全局特征向量

### 2. 特征提取策略
- **全局特征**: 使用CLS token表示整个图像/目标区域
- **目标裁剪**: 根据边界框裁剪目标区域
- **标准化**: 使用ImageNet标准化参数

### 3. 分析方法
- **降维可视化**: PCA和t-SNE降维
- **相似性度量**: 余弦相似性计算
- **聚类分析**: 基于特征的目标聚类

## 实验结果

运行程序后会生成以下结果：

1. **检测结果** (`detection_results.png`)
   - 显示原图和检测到的目标边界框
   - 标注目标类别
   - 展示裁剪的目标区域

2. **特征分析** (`feature_analysis.png`)
   - PCA降维可视化（显示主成分解释方差）
   - t-SNE降维可视化（展示非线性结构）
   - 特征向量分布图
   - 目标类别统计图

3. **相似性矩阵** (`similarity_matrix.png`)
   - 特征间余弦相似性热图
   - 类别间相似性数值分析

## 性能指标
- **特征维度**: 768维
- **处理速度**: CPU模式下每张图像约1-2秒
- **内存占用**: 模型加载约400MB
- **准确性**: 基于预训练权重，具有良好的特征表示能力


