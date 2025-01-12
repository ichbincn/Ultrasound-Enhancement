# ACP/__init__.py
# 初始化 ACP 模块，导入所有必要的子模块

# 导入模型相关模块
from .models.classifier import Classifier
from .models.transformer_classifier import TransformerClassifier
from .models.feature_extractor import FeatureExtractor

# 导入数据集相关模块
from .datasets.medical_dataset import MedicalDataset
from .datasets.augmentations import get_augmentations

# 导入损失函数
from .losses.mse_loss import mse_loss
from .losses.ssim_loss import ssim_loss
from .losses.combined_loss import combined_loss


# 为用户提供的主要接口
__all__ = [
    # 模型
    "Classifier",
    "TransformerClassifier",
    "FeatureExtractor",
    # 数据集
    "MedicalDataset",
    "get_augmentations",
    # 损失函数
    "mse_loss",
    "ssim_loss",
    "combined_loss",

]
