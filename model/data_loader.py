"""
Legacy загрузчик для 7-классового режима (FER-2013).
В B2 версии (3 класса) используется train_fer2013_b2.py.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import os


class FER2013Dataset(Dataset):
    """
    Dataset для работы с FER-2013
    Формат CSV: emotion, pixels, Usage
    """
    
    def __init__(self, csv_file=None, transform=None, usage='Training'):
        """
        Args:
            csv_file: Путь к CSV файлу с данными
            transform: Трансформации для аугментации
            usage: 'Training', 'PublicTest', или 'PrivateTest'
        """
        self.transform = transform
        self.usage = usage
        
        # Маппинг эмоций
        self.emotion_labels = {
            0: 'Злость',
            1: 'Отвращение',
            2: 'Страх',
            3: 'Счастье',
            4: 'Грусть',
            5: 'Удивление',
            6: 'Нейтральное'
        }
        
        # Если CSV файл предоставлен, загружаем данные
        if csv_file and os.path.exists(csv_file):
            self.load_from_csv(csv_file)
        else:
            # Генерируем синтетические данные для демонстрации
            print("CSV файл не найден. Создание демонстрационного датасета...")
            self.create_synthetic_dataset()
    
    def load_from_csv(self, csv_file):
        """
        Загрузка данных из CSV файла FER-2013
        """
        print(f"Загрузка данных из {csv_file}...")
        df = pd.read_csv(csv_file)
        
        # Фильтрация по типу использования
        if self.usage:
            df = df[df['Usage'] == self.usage]
        
        self.data = []
        self.labels = []
        
        for idx, row in df.iterrows():
            # Преобразование строки пикселей в массив
            pixels = np.array([int(p) for p in row['pixels'].split()], dtype=np.uint8)
            image = pixels.reshape(48, 48)
            
            self.data.append(image)
            self.labels.append(row['emotion'])
        
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        
        print(f"Загружено {len(self.data)} изображений")
        print(f"Распределение классов: {np.bincount(self.labels)}")
    
    def create_synthetic_dataset(self):
        """
        Создание синтетического датасета для демонстрации
        """
        n_samples_per_class = 100 if self.usage == 'Training' else 20
        n_classes = 7
        
        self.data = []
        self.labels = []
        
        for emotion in range(n_classes):
            for _ in range(n_samples_per_class):
                # Создание случайного изображения с некоторыми паттернами
                img = np.random.randint(0, 256, (48, 48), dtype=np.uint8)
                
                # Добавление простых паттернов для различных эмоций
                if emotion == 3:  # Счастье - добавим "улыбку"
                    cv2.ellipse(img, (24, 30), (15, 10), 0, 0, 180, 255, 2)
                elif emotion == 4:  # Грусть - добавим "грустный рот"
                    cv2.ellipse(img, (24, 35), (15, 10), 0, 180, 360, 255, 2)
                
                self.data.append(img)
                self.labels.append(emotion)
        
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        
        print(f"Создано {len(self.data)} синтетических изображений")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        
        # Преобразование в PIL Image для трансформаций
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        else:
            # Базовое преобразование
            image = transforms.ToTensor()(image)
        
        return image, label
    
    def get_emotion_name(self, label):
        """Получить название эмоции по метке"""
        return self.emotion_labels[label]


class ImageDataset(Dataset):
    """
    Dataset для загрузки изображений из папки
    """
    
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.images = []
        
        # Загрузка всех изображений из папки
        for filename in os.listdir(image_folder):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                self.images.append(os.path.join(image_folder, filename))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('L')  # Grayscale
        
        if self.transform:
            image = self.transform(image)
        
        return image, image_path


def get_data_transforms(augment=True):
    """
    Получение трансформаций для обучения и валидации
    
    Args:
        augment: Применять ли аугментацию данных
        
    Returns:
        train_transform, test_transform
    """
    
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomResizedCrop(48, scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    test_transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    return train_transform, test_transform


def create_data_loaders(csv_file=None, batch_size=64, num_workers=2):
    """
    Создание DataLoader для обучения и валидации
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    train_transform, test_transform = get_data_transforms(augment=True)
    
    # Создание датасетов
    train_dataset = FER2013Dataset(
        csv_file=csv_file,
        transform=train_transform,
        usage='Training'
    )
    
    val_dataset = FER2013Dataset(
        csv_file=csv_file,
        transform=test_transform,
        usage='PublicTest'
    )
    
    test_dataset = FER2013Dataset(
        csv_file=csv_file,
        transform=test_transform,
        usage='PrivateTest'
    )
    
    # Создание DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def preprocess_single_image(image_path):
    """
    Предобработка одного изображения для предсказания
    
    Args:
        image_path: Путь к изображению или numpy array
        
    Returns:
        Preprocessed tensor
    """
    # Загрузка изображения
    if isinstance(image_path, str):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        image = image_path
    
    # Обнаружение лица (опционально)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    faces = face_cascade.detectMultiScale(image, 1.3, 5)
    
    # Если лицо найдено, обрезаем изображение
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        image = image[y:y+h, x:x+w]
    
    # Изменение размера
    image = cv2.resize(image, (48, 48))
    
    # Нормализация
    image = image.astype(np.float32) / 255.0
    image = (image - 0.5) / 0.5
    
    # Преобразование в тензор
    image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    
    return image


if __name__ == "__main__":
    # Тестирование загрузчика данных
    print("Тестирование загрузчика данных...")
    
    train_loader, val_loader, test_loader = create_data_loaders(
        csv_file=None,  # Используем синтетические данные
        batch_size=32
    )
    
    # Получение одного батча
    images, labels = next(iter(train_loader))
    print(f"Размер батча изображений: {images.shape}")
    print(f"Размер батча меток: {labels.shape}")
    print(f"Уникальные метки: {torch.unique(labels)}")
