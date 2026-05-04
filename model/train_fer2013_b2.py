import argparse
import os
import random
from datetime import datetime

import numpy as np
import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from cnn_architecture import EmotionCNN


class EmotionFolderDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("L")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_class_dir(root_dir: str, class_name: str):
    direct = os.path.join(root_dir, class_name)
    if os.path.isdir(direct):
        return direct

    if os.path.isdir(root_dir):
        for d in os.listdir(root_dir):
            cand = os.path.join(root_dir, d)
            if os.path.isdir(cand) and d.lower() == class_name.lower():
                return cand

    return None


def list_class_images(root_dir: str, class_name: str):
    class_dir = _resolve_class_dir(root_dir, class_name)
    if class_dir is None:
        return []

    files = []
    for fn in os.listdir(class_dir):
        low = fn.lower()
        if low.endswith(".png") or low.endswith(".jpg") or low.endswith(".jpeg"):
            files.append(os.path.join(class_dir, fn))
    files.sort()
    return files


def sample_files(files, rng: random.Random, limit):
    files = files[:]
    rng.shuffle(files)
    if limit is None:
        return files
    if limit <= 0:
        return files
    return files[:limit]


def build_split_from_dirs(train_dir, val_dir, classes, seed, max_train_per_class, max_val_per_class):
    rng = random.Random(seed)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    train_items = []
    val_items = []

    for c in classes:
        train_files = list_class_images(train_dir, c)
        val_files = list_class_images(val_dir, c)

        train_files = sample_files(train_files, rng, max_train_per_class)
        val_files = sample_files(val_files, rng, max_val_per_class)

        train_items.extend([(p, class_to_idx[c]) for p in train_files])
        val_items.extend([(p, class_to_idx[c]) for p in val_files])

    return train_items, val_items, class_to_idx


def build_split_by_fraction(train_dir, classes, seed, max_train_per_class, max_val_per_class, val_fraction):
    rng = random.Random(seed)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    train_items = []
    val_items = []

    for c in classes:
        files = list_class_images(train_dir, c)
        if not files:
            continue

        files = sample_files(files, rng, None)

        n_total = len(files)
        n_val_default = max(1, int(round(n_total * val_fraction)))
        n_val = n_val_default
        if max_val_per_class is not None:
            n_val = min(n_val, max_val_per_class)

        remaining = files[n_val:]
        if max_train_per_class is not None:
            remaining = remaining[:max_train_per_class]

        val_slice = files[:n_val]
        train_slice = remaining

        train_items.extend([(p, class_to_idx[c]) for p in train_slice])
        val_items.extend([(p, class_to_idx[c]) for p in val_slice])

    return train_items, val_items, class_to_idx


def build_transforms(augment: bool):
    if augment:
        train_t = transforms.Compose(
            [
                transforms.Resize((48, 48)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.95, 1.05)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
    else:
        train_t = transforms.Compose(
            [
                transforms.Resize((48, 48)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    eval_t = transforms.Compose(
        [
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    return train_t, eval_t


def make_weighted_sampler(items, num_classes):
    labels = [lbl for _, lbl in items]
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    class_weights = 1.0 / counts
    sample_weights = [class_weights[lbl] for lbl in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def accuracy_from_logits(logits, labels):
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()


def disable_dropout(model: nn.Module):
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.p = 0.0


def recalibrate_batchnorm(model: nn.Module, loader: DataLoader, device: str):
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.eval()
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.train()

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            _ = model(x)


def run_epoch(model, loader, device, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_acc = 0.0
    total_n = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy_from_logits(logits, labels) * bs
        total_n += bs

    return total_loss / max(1, total_n), total_acc / max(1, total_n)


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_n = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_acc += accuracy_from_logits(logits, labels) * bs
            total_n += bs

    return total_loss / max(1, total_n), total_acc / max(1, total_n)


def save_checkpoint(path, model, class_to_idx, best_val_acc, args, epoch):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "labels": class_to_idx,
            "best_val_accuracy": float(best_val_acc),
            "epoch": int(epoch),
            "args": vars(args),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        path,
    )


def save_training_history(out_dir, history):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "training_history.json")
    with open(path, "w") as f:
        json.dump(history, f, indent=4)
    print("HISTORY:", path)


def plot_training_history(out_dir, history):
    os.makedirs(out_dir, exist_ok=True)

    train_losses = history["train_losses"]
    val_losses = history["val_losses"]
    train_accs = [a * 100 for a in history["train_accuracies"]]
    val_accs = [a * 100 for a in history["val_accuracies"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(train_losses, label="Train Loss", linewidth=2)
    axes[0].plot(val_losses, label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(train_accs, label="Train Acc", linewidth=2)
    axes[1].plot(val_accs, label="Val Acc", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "training_history.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print("PLOT:", path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="data/fer2013")
    parser.add_argument("--train_dir", type=str, default=None)
    parser.add_argument("--val_dir", type=str, default=None)

    parser.add_argument("--classes", type=str, default="happy,sad,neutral,angry,surprise")
    parser.add_argument("--val_fraction", type=float, default=0.15)

    parser.add_argument("--max_train_per_class", type=int, default=200)
    parser.add_argument("--max_val_per_class", type=int, default=60)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--no_balance", action="store_true")

    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--backup_existing", action="store_true")

    parser.add_argument("--val_same_as_train", action="store_true")
    parser.add_argument("--dropout_off", action="store_true")
    parser.add_argument("--recalibrate_bn", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    if len(classes) < 2:
        raise ValueError("Нужно минимум 2 класса")

    max_train = args.max_train_per_class if args.max_train_per_class > 0 else None
    max_val = args.max_val_per_class if args.max_val_per_class > 0 else None

    train_dir = args.train_dir
    val_dir = args.val_dir
    if train_dir is None:
        train_dir = os.path.join(args.data_root, "train")

    if val_dir is None:
        candidates = [
            os.path.join(args.data_root, "test"),
            os.path.join(args.data_root, "val"),
            os.path.join(args.data_root, "validation"),
        ]
        val_dir = next((p for p in candidates if os.path.isdir(p)), None)

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Не найдена папка train_dir: {train_dir}")

    if val_dir is not None and os.path.isdir(val_dir):
        train_items, val_items, class_to_idx = build_split_from_dirs(
            train_dir=train_dir,
            val_dir=val_dir,
            classes=classes,
            seed=args.seed,
            max_train_per_class=max_train,
            max_val_per_class=max_val,
        )
    else:
        train_items, val_items, class_to_idx = build_split_by_fraction(
            train_dir=train_dir,
            classes=classes,
            seed=args.seed,
            max_train_per_class=max_train,
            max_val_per_class=max_val,
            val_fraction=args.val_fraction,
        )

    if args.val_same_as_train:
        val_items = train_items[:]

    if len(train_items) == 0 or len(val_items) == 0:
        raise RuntimeError("Пустые train/val сплиты: проверь структуру датасета и названия папок классов")

    num_classes = len(classes)
    dist_train = np.bincount([l for _, l in train_items], minlength=num_classes)
    dist_val = np.bincount([l for _, l in val_items], minlength=num_classes)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("DEVICE:", device)
    print("TRAIN DIR:", train_dir)
    print("VAL DIR:", val_dir if val_dir is not None else "(split from train)")
    print("CLASSES:", classes)
    print("TRAIN SAMPLES:", len(train_items), "PER CLASS:", dist_train.tolist())
    print("VAL SAMPLES:", len(val_items), "PER CLASS:", dist_val.tolist())

    train_t, eval_t = build_transforms(augment=args.augment)

    train_ds = EmotionFolderDataset(train_items, transform=train_t)
    val_ds = EmotionFolderDataset(val_items, transform=eval_t)

    sampler = None
    shuffle = True
    if not args.no_balance:
        sampler = make_weighted_sampler(train_items, num_classes=num_classes)
        shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = EmotionCNN(num_classes=num_classes).to(device)

    if args.dropout_off:
        disable_dropout(model)

    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.output is None:
        args.output = os.path.join(script_dir, "checkpoints", "fer2013_5c", "best_model_5c.pth")

    if args.backup_existing and os.path.exists(args.output):
        out_dir = os.path.dirname(args.output)
        base = os.path.basename(args.output)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(out_dir, f"{base}.bak_{ts}")
        os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "rb") as src:
            with open(backup_path, "wb") as dst:
                dst.write(src.read())
        print("BACKUP:", backup_path)

    best_val_acc = -1.0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, device, criterion, optimizer=optimizer)

        if args.recalibrate_bn:
            recalibrate_batchnorm(model, train_loader, device)

        val_loss, val_acc = evaluate(model, val_loader, device, criterion)

        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))
        train_accuracies.append(float(train_acc))
        val_accuracies.append(float(val_acc))

        print(
            f"EPOCH {epoch:03d}/{args.epochs:03d} "
            f"| train loss {train_loss:.4f} acc {train_acc*100:.2f}% "
            f"| val loss {val_loss:.4f} acc {val_acc*100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(args.output, model, class_to_idx, best_val_acc, args, epoch)
            print("SAVED:", args.output)

    out_dir = os.path.dirname(args.output)
    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
        "best_val_accuracy": float(best_val_acc),
    }
    save_training_history(out_dir, history)
    plot_training_history(out_dir, history)

    print("BEST VAL ACC:", round(best_val_acc * 100, 2), "%")


if __name__ == "__main__":
    main()
