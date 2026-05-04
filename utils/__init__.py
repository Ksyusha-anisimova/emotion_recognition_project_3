"""
Утилиты для проекта распознавания эмоций
"""

from .visualization import (
    plot_confusion_matrix,
    visualize_feature_maps,
    plot_emotion_distribution,
    visualize_predictions,
    plot_model_comparison,
    create_emotion_heatmap
)

__all__ = [
    'plot_confusion_matrix',
    'visualize_feature_maps',
    'plot_emotion_distribution',
    'visualize_predictions',
    'plot_model_comparison',
    'create_emotion_heatmap'
]

__version__ = '1.0.0'
