#!/usr/bin/env python3
"""
Скрипт быстрого запуска проекта распознавания эмоций
"""

import os
import sys
import subprocess
import argparse


MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    'model',
    'checkpoints',
    'best_model_b2.pth'
)


def print_banner():
    """Вывод баннера проекта"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║     🤖  СИСТЕМА РАСПОЗНАВАНИЯ ЭМОЦИЙ  🤖                    ║
    ║                                                              ║
    ║     Дипломный проект по глубокому обучению                   ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_requirements():
    """Корректная проверка установленных зависимостей"""
    print("\n📦 Проверка зависимостей...")

    import_map = {
        "torch": "torch",
        "torchvision": "torchvision",
        "flask": "flask",
        "opencv-python": "cv2",
        "numpy": "numpy",
        "matplotlib": "matplotlib",
        "Pillow": "PIL",
    }

    missing = []

    for pkg, module in import_map.items():
        try:
            __import__(module)
            print(f"  ✓ {pkg}")
        except Exception:
            print(f"  ✗ {pkg} (ошибка импорта)")
            missing.append(pkg)

    if missing:
        print("\n⚠️  Обнаружены проблемы с пакетами:")
        print("  pip install " + " ".join(missing))
        return False

    print("\n✓ Все зависимости корректно установлены!")
    return True


def create_demo_model():
    """Создание демонстрационной модели B2 (ТОЛЬКО если нет обученной)"""
    print("\n🔨 Создание демонстрационной модели...")

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'model'))

    try:
        import torch
        from model.cnn_architecture import EmotionCNN

        model = EmotionCNN(num_classes=3)

        checkpoint_dir = os.path.dirname(MODEL_PATH)
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            'epoch': 0,
            'model_state_dict': model.state_dict(),
            'accuracy': 0.0,
            'labels': {
                "happy": 0,
                "sad": 1,
                "neutral": 2,
            },
        }

        torch.save(checkpoint, MODEL_PATH)

        print(f"✓ Демонстрационная модель создана: {MODEL_PATH}")
        return True

    except Exception as e:
        print(f"Ошибка при создании демо-модели: {e}")
        return False


def ensure_model_exists():
    """Проверка наличия модели, без затирания"""
    if os.path.exists(MODEL_PATH):
        print(f"\n✓ Найдена обученная B2 модель: {MODEL_PATH}")
        print("✓ Демо-модель НЕ создаётся")
        return True
    else:
        print("\n⚠ Обученная B2 модель не найдена")
        return create_demo_model()


def run_web_app():
    """Запуск веб-приложения"""
    print("\n🌐 ЗАПУСК ВЕБ-ПРИЛОЖЕНИЯ")
    print("=" * 70)

    web_app_path = os.path.join(
        os.path.dirname(__file__),
        'web_app',
        'app.py'
    )

    if not os.path.exists(web_app_path):
        print(f"Файл {web_app_path} не найден!")
        return False

    print("\n🚀 Запуск Flask сервера...")
    print("📍 Приложение будет доступно по адресу: http://localhost:5000")
    print("\nДля остановки нажмите Ctrl+C\n")

    try:
        subprocess.run(['python', web_app_path])
    except KeyboardInterrupt:
        print("\n\nПриложение остановлено")
    except Exception as e:
        print(f"\nОшибка: {e}")
        return False

    return True


def train_model(
    epochs=10,
    batch_size=64,
    train_dir=None,
    val_dir=None,
    augment=False,
    no_balance=False,
):
    """Запуск обучения модели (B2: 3 класса)"""
    print("\n🎓 ЗАПУСК ОБУЧЕНИЯ МОДЕЛИ")
    print("=" * 70)

    try:
        script_path = os.path.join(
            os.path.dirname(__file__),
            'model',
            'train_fer2013_b2.py'
        )

        cmd = [
            sys.executable,
            script_path,
            '--epochs', str(epochs),
            '--batch_size', str(batch_size),
        ]

        if train_dir:
            cmd += ['--train_dir', train_dir]
        if val_dir:
            cmd += ['--val_dir', val_dir]
        if augment:
            cmd += ['--augment']
        if no_balance:
            cmd += ['--no_balance']

        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"\nОшибка при обучении: {e}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Система распознавания эмоций - Дипломный проект'
    )

    parser.add_argument(
        'command',
        choices=['train', 'run', 'demo', 'check'],
        help='Команда для выполнения'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Количество эпох обучения (по умолчанию: 10)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Размер батча (по умолчанию: 64)'
    )

    parser.add_argument(
        '--train-dir',
        type=str,
        default='data/custom_train_b2',
        help='Путь к папке train с классами (по умолчанию: data/custom_train_b2)'
    )

    parser.add_argument(
        '--val-dir',
        type=str,
        default=None,
        help='Путь к папке val/test (если нет, берётся сплит из train)'
    )

    parser.add_argument(
        '--augment',
        action='store_true',
        help='Включить аугментации'
    )

    parser.add_argument(
        '--no-balance',
        action='store_true',
        help='Отключить балансировку классов'
    )

    args = parser.parse_args()

    print_banner()

    if args.command == 'check':
        check_requirements()

    elif args.command == 'demo':
        if check_requirements():
            ensure_model_exists()
            run_web_app()

    elif args.command == 'train':
        if check_requirements():
            train_model(
                epochs=args.epochs,
                batch_size=args.batch_size,
                train_dir=args.train_dir,
                val_dir=args.val_dir,
                augment=args.augment,
                no_balance=args.no_balance,
            )

    elif args.command == 'run':
        if check_requirements():
            ensure_model_exists()
            run_web_app()


if __name__ == '__main__':
    main()
