"""
Модуль модели распознавания эмоций
"""

from .cnn_architecture import EmotionCNN
from .data_loader import (
    FER2013Dataset,
    get_data_transforms,
    create_data_loaders,
    preprocess_single_image
)
from .train import EmotionTrainer

__all__ = [
    'EmotionCNN',
    'FER2013Dataset',
    'get_data_transforms',
    'create_data_loaders',
    'preprocess_single_image',
    'EmotionTrainer'
]

__version__ = '1.0.0'
