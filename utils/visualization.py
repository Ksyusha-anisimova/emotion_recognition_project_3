"""
Утилиты для визуализации результатов и метрик модели
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
import torch


def plot_confusion_matrix(confusion_matrix, class_names, save_path=None):
    """
    Визуализация матрицы ошибок
    """
    plt.figure(figsize=(10, 8))
    
    # Нормализация матрицы
    cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Value'}
    )
    
    plt.xlabel('Предсказанная эмоция', fontsize=12)
    plt.ylabel('Истинная эмоция', fontsize=12)
    plt.title('Матрица ошибок (Confusion Matrix)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Матрица ошибок сохранена в {save_path}")
    
    plt.show()


def visualize_feature_maps(model, image_tensor, save_path=None):
    """
    Визуализация карт признаков на разных слоях сети
    """
    model.eval()
    
    with torch.no_grad():
        # Получение промежуточных признаков
        features = model.get_feature_maps(image_tensor)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle('Карты признаков на разных слоях', fontsize=16, fontweight='bold')
    
    layer_names = ['conv1', 'pool1', 'conv3', 'pool2']
    
    for idx, (ax, layer_name) in enumerate(zip(axes.flat, layer_names)):
        if layer_name in features:
            feature_map = features[layer_name][0].cpu().numpy()
            
            # Усреднение по каналам
            feature_avg = np.mean(feature_map, axis=0)
            
            im = ax.imshow(feature_avg, cmap='viridis')
            ax.set_title(f'Слой: {layer_name}', fontsize=12, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Карты признаков сохранены в {save_path}")
    
    plt.show()


def plot_emotion_distribution(dataset, save_path=None):
    """
    Визуализация распределения эмоций в датасете
    """
    emotion_labels = {
        0: 'Злость',
        1: 'Отвращение',
        2: 'Страх',
        3: 'Счастье',
        4: 'Грусть',
        5: 'Удивление',
        6: 'Нейтральное'
    }
    
    # Подсчет эмоций
    labels = [label for _, label in dataset]
    unique, counts = np.unique(labels, return_counts=True)
    
    # Создание графика
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#FF6B6B', '#95E1D3', '#A8E6CF', '#FFD93D', '#6BCB77', '#FF6B9D', '#C7CEEA']
    
    bars = ax.bar([emotion_labels[i] for i in unique], counts, color=colors)
    
    # Добавление значений на столбцы
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Эмоция', fontsize=12)
    ax.set_ylabel('Количество изображений', fontsize=12)
    ax.set_title('Распределение эмоций в датасете', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График распределения сохранен в {save_path}")
    
    plt.show()


def visualize_predictions(model, dataloader, device, num_samples=16):
    """
    Визуализация предсказаний модели на случайных изображениях
    """
    model.eval()
    
    emotion_labels = {
        0: 'Злость',
        1: 'Отвращение',
        2: 'Страх',
        3: 'Счастье',
        4: 'Грусть',
        5: 'Удивление',
        6: 'Нейтральное'
    }
    
    # Получение батча
    images, labels = next(iter(dataloader))
    images = images[:num_samples].to(device)
    labels = labels[:num_samples]
    
    # Предсказания
    with torch.no_grad():
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
    
    # Визуализация
    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    fig.suptitle('Предсказания модели', fontsize=16, fontweight='bold')
    
    for idx, ax in enumerate(axes.flat):
        if idx < num_samples:
            # Денормализация изображения
            img = images[idx].cpu().squeeze().numpy()
            img = (img * 0.5) + 0.5  # Обратная нормализация
            img = np.clip(img, 0, 1)
            
            true_label = emotion_labels[labels[idx].item()]
            pred_label = emotion_labels[predictions[idx].item()]
            
            # Цвет рамки в зависимости от правильности
            color = 'green' if labels[idx] == predictions[idx] else 'red'
            
            ax.imshow(img, cmap='gray')
            ax.set_title(f'Истина: {true_label}\nПредсказание: {pred_label}',
                        fontsize=10, color=color, fontweight='bold')
            ax.axis('off')
            
            # Рамка
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
    
    plt.tight_layout()
    plt.show()


def plot_model_comparison(history1, history2, label1='Model 1', label2='Model 2'):
    """
    Сравнение двух моделей по метрикам
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    axes[0].plot(history1['val_losses'], label=f'{label1} Val Loss', linewidth=2)
    axes[0].plot(history2['val_losses'], label=f'{label2} Val Loss', linewidth=2)
    axes[0].set_xlabel('Эпоха', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Сравнение функции потерь', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history1['val_accuracies'], label=f'{label1} Val Accuracy', linewidth=2)
    axes[1].plot(history2['val_accuracies'], label=f'{label2} Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Эпоха', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Сравнение точности', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def create_emotion_heatmap(image_path, model, device):
    """
    Создание тепловой карты активации (Grad-CAM)
    """
    # Загрузка изображения
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_resized = cv2.resize(image, (48, 48))
    
    # Предобработка
    img_tensor = torch.from_numpy(image_resized).float().unsqueeze(0).unsqueeze(0)
    img_tensor = (img_tensor / 255.0 - 0.5) / 0.5
    img_tensor = img_tensor.to(device)
    img_tensor.requires_grad = True
    
    # Прямой проход
    model.eval()
    output = model(img_tensor)
    
    # Получение градиентов
    model.zero_grad()
    class_idx = output.argmax().item()
    output[0, class_idx].backward()
    
    # Визуализация
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Оригинальное изображение
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Оригинал', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Градиенты
    gradients = img_tensor.grad.cpu().squeeze().numpy()
    axes[1].imshow(np.abs(gradients), cmap='hot')
    axes[1].set_title('Градиенты', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Наложение
    heatmap = cv2.resize(np.abs(gradients), (image.shape[1], image.shape[0]))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    
    # Преобразование в цвет
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Наложение на оригинал
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    superimposed = cv2.addWeighted(image_rgb, 0.6, heatmap_colored, 0.4, 0)
    
    axes[2].imshow(superimposed)
    axes[2].set_title('Области внимания', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Модуль визуализации загружен успешно")
