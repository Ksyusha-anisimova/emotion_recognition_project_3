"""
Архитектура сверточной нейронной сети для распознавания эмоций.
Реализация включает объяснение каждого слоя и его роли.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmotionCNN(nn.Module):
    """
    Собственная архитектура CNN для распознавания эмоций.

    B2 версия (основная, 3 класса):
    0: Счастье (Happy)
    1: Грусть (Sad)
    2: Нейтральное (Neutral)

    Legacy версия (7 классов) поддерживается при num_classes=7.
    """

    def __init__(self, num_classes=3):
        super(EmotionCNN, self).__init__()
        
        # Блок 1: Извлечение низкоуровневых признаков (края, текстуры)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(0.25)
        
        # Блок 2: Извлечение среднеуровневых признаков (формы, паттерны)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(0.25)
        
        # Блок 3: Извлечение высокоуровневых признаков (сложные паттерны лица)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout2d(0.25)
        
        # Блок 4: Глубокие признаки
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout4 = nn.Dropout2d(0.25)
        
        # Полносвязные слои для классификации
        self.fc1 = nn.Linear(512 * 3 * 3, 512)  # Для входа 48x48
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout5 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dropout6 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        """
        Прямой проход через сеть
        
        Args:
            x: Входное изображение [batch_size, 1, 48, 48]
            
        Returns:
            Логиты для каждого класса [batch_size, num_classes]
        """
        # Блок 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Блок 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Блок 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Блок 4
        x = F.relu(self.bn7(self.conv7(x)))
        x = self.pool4(x)
        x = self.dropout4(x)
        
        # Преобразование в вектор
        x = x.view(x.size(0), -1)
        
        # Полносвязные слои
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout5(x)
        
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout6(x)
        
        x = self.fc3(x)
        
        return x
    
    def get_feature_maps(self, x):
        """
        Получение промежуточных признаков для визуализации
        """
        features = {}
        
        x = F.relu(self.bn1(self.conv1(x)))
        features['conv1'] = x
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        features['pool1'] = x
        
        x = F.relu(self.bn3(self.conv3(x)))
        features['conv3'] = x
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        features['pool2'] = x
        
        return features


class CustomConvLayer:
    """
    Демонстрация работы сверточного слоя "под капотом"
    Для дипломной работы: показывает понимание принципов работы CNN
    """
    
    @staticmethod
    def conv2d_manual(input_tensor, kernel, stride=1, padding=0):
        """
        Ручная реализация 2D свертки для демонстрации принципа работы
        
        Args:
            input_tensor: Входной тензор [H, W]
            kernel: Ядро свертки [K, K]
            stride: Шаг свертки
            padding: Количество пикселей для дополнения
            
        Returns:
            Результат свертки
        """
        import numpy as np
        
        # Добавление padding
        if padding > 0:
            input_tensor = np.pad(input_tensor, padding, mode='constant')
        
        H, W = input_tensor.shape
        K = kernel.shape[0]
        
        # Вычисление размера выхода
        out_h = (H - K) // stride + 1
        out_w = (W - K) // stride + 1
        
        output = np.zeros((out_h, out_w))
        
        # Применение свертки
        for i in range(0, out_h):
            for j in range(0, out_w):
                h_start = i * stride
                w_start = j * stride
                
                # Элементное умножение и сумма
                region = input_tensor[h_start:h_start+K, w_start:w_start+K]
                output[i, j] = np.sum(region * kernel)
        
        return output
    
    @staticmethod
    def explain_convolution():
        """
        Объяснение работы свертки для документации
        """
        explanation = """
        ПРИНЦИП РАБОТЫ СВЕРТОЧНОГО СЛОЯ:
        
        1. Свертка (Convolution):
           - Скользящее окно (kernel) проходит по изображению
           - На каждой позиции вычисляется скалярное произведение
           - Результат формирует карту признаков (feature map)
        
        2. Что извлекает свертка:
           - Первые слои: края, линии, простые текстуры
           - Средние слои: формы, паттерны (глаза, рот, нос)
           - Глубокие слои: сложные признаки (выражения лица)
        
        3. Batch Normalization:
           - Нормализует активации между слоями
           - Ускоряет обучение и стабилизирует процесс
        
        4. MaxPooling:
           - Уменьшает размерность
           - Выбирает наиболее важные признаки
           - Обеспечивает инвариантность к небольшим сдвигам
        
        5. Dropout:
           - Случайно отключает нейроны во время обучения
           - Предотвращает переобучение
        """
        return explanation


def count_parameters(model):
    """
    Подсчет количества обучаемых параметров
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model):
    """
    Вывод сводной информации о модели
    """
    print("=" * 70)
    print("АРХИТЕКТУРА МОДЕЛИ")
    print("=" * 70)
    print(model)
    print("=" * 70)
    print(f"Общее количество параметров: {count_parameters(model):,}")
    print("=" * 70)


if __name__ == "__main__":
    # Тестирование модели
    model = EmotionCNN(num_classes=3)
    model_summary(model)
    
    # Тест прямого прохода
    dummy_input = torch.randn(1, 1, 48, 48)
    output = model(dummy_input)
    print(f"\nРазмер входа: {dummy_input.shape}")
    print(f"Размер выхода: {output.shape}")
    print(f"Выходные логиты: {output}")
