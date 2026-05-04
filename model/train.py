"""
Legacy модуль обучения 7-классовой модели.
В B2 версии (3 класса) используется model/train_fer2013_b2.py.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from datetime import datetime

from cnn_architecture import EmotionCNN
from data_loader import create_data_loaders


class EmotionTrainer:
    """
    Класс для обучения модели распознавания эмоций
    """
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        
        # История обучения
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Лучшая точность
        self.best_val_accuracy = 0.0
        
        print(f"Используется устройство: {device}")
    
    def train_epoch(self, train_loader, criterion, optimizer):
        """
        Обучение на одной эпохе
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc='Обучение')
        
        for images, labels in progress_bar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Обнуление градиентов
            optimizer.zero_grad()
            
            # Прямой проход
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            
            # Обратный проход
            loss.backward()
            optimizer.step()
            
            # Статистика
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Обновление прогресс-бара
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader, criterion):
        """
        Валидация модели
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Матрица ошибок
        confusion_matrix = np.zeros((7, 7), dtype=int)
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Валидация'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Обновление матрицы ошибок
                for t, p in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
                    confusion_matrix[t, p] += 1
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc, confusion_matrix
    
    def train(self, train_loader, val_loader, num_epochs=50, 
              learning_rate=0.001, save_dir='checkpoints'):
        """
        Полный цикл обучения
        """
        # Создание директории для чекпоинтов
        os.makedirs(save_dir, exist_ok=True)
        
        # Функция потерь и оптимизатор
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Планировщик learning rate
        scheduler = ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        print("\n" + "="*70)
        print("НАЧАЛО ОБУЧЕНИЯ")
        print("="*70)
        
        for epoch in range(num_epochs):
            print(f"\nЭпоха {epoch+1}/{num_epochs}")
            print("-" * 70)
            
            # Обучение
            train_loss, train_acc = self.train_epoch(
                train_loader, criterion, optimizer
            )
            
            # Валидация
            val_loss, val_acc, conf_matrix = self.validate(
                val_loader, criterion
            )
            
            # Сохранение истории
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Обновление learning rate
            scheduler.step(val_acc)
            
            # Вывод результатов
            print(f"\nРезультаты эпохи {epoch+1}:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Сохранение лучшей модели
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                self.save_model(
                    os.path.join(save_dir, 'best_model.pth'),
                    epoch, val_acc, conf_matrix
                )
                print(f"  ✓ Новая лучшая модель сохранена! (Acc: {val_acc:.2f}%)")
            
            # Периодическое сохранение
            if (epoch + 1) % 10 == 0:
                self.save_model(
                    os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'),
                    epoch, val_acc, conf_matrix
                )
        
        print("\n" + "="*70)
        print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
        print(f"Лучшая точность на валидации: {self.best_val_accuracy:.2f}%")
        print("="*70)
        
        # Сохранение истории обучения
        self.save_training_history(save_dir)
        
        # Визуализация результатов
        self.plot_training_history(save_dir)
        
        return self.best_val_accuracy
    
    def save_model(self, filepath, epoch, accuracy, confusion_matrix=None):
        """
        Сохранение модели и метаданных
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'accuracy': accuracy,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'confusion_matrix': confusion_matrix.tolist() if confusion_matrix is not None else None,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        torch.save(checkpoint, filepath)
    
    def load_model(self, filepath):
        """
        Загрузка модели
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Восстановление истории обучения
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            self.train_accuracies = checkpoint['train_accuracies']
            self.val_accuracies = checkpoint['val_accuracies']
        
        print(f"Модель загружена из {filepath}")
        print(f"Эпоха: {checkpoint['epoch']}, Точность: {checkpoint['accuracy']:.2f}%")
        
        return checkpoint
    
    def save_training_history(self, save_dir):
        """
        Сохранение истории обучения в JSON
        """
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_val_accuracy': self.best_val_accuracy
        }
        
        filepath = os.path.join(save_dir, 'training_history.json')
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=4)
        
        print(f"История обучения сохранена в {filepath}")
    
    def plot_training_history(self, save_dir):
        """
        Визуализация процесса обучения
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # График функции потерь
        axes[0].plot(self.train_losses, label='Train Loss', linewidth=2)
        axes[0].plot(self.val_losses, label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Эпоха', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Функция потерь', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # График точности
        axes[1].plot(self.train_accuracies, label='Train Accuracy', linewidth=2)
        axes[1].plot(self.val_accuracies, label='Val Accuracy', linewidth=2)
        axes[1].set_xlabel('Эпоха', fontsize=12)
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].set_title('Точность модели', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(save_dir, 'training_history.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"График обучения сохранен в {filepath}")
        plt.close()


def main():
    """
    Основная функция для запуска обучения
    """
    # Параметры обучения
    BATCH_SIZE = 64
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.001
    
    # Создание модели
    model = EmotionCNN(num_classes=7)
    
    # Создание загрузчиков данных
    train_loader, val_loader, test_loader = create_data_loaders(
        csv_file=None,  # Путь к FER-2013 CSV или None для синтетических данных
        batch_size=BATCH_SIZE
    )
    
    # Создание тренера
    trainer = EmotionTrainer(model)
    
    # Обучение
    best_accuracy = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        save_dir='checkpoints'
    )
    
    print(f"\nФинальная лучшая точность: {best_accuracy:.2f}%")


if __name__ == "__main__":
    main()
