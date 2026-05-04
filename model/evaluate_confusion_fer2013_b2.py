import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from cnn_architecture import EmotionCNN
from train_fer2013_b2 import EmotionFolderDataset, build_transforms


def load_checkpoint(checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=device)


def build_test_dataset(data_root, classes):
    test_dir = os.path.join(data_root, "test")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    items = []
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for c in classes:
        class_dir = os.path.join(test_dir, c)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Class directory not found: {class_dir}")
        for fn in sorted(os.listdir(class_dir)):
            low = fn.lower()
            if low.endswith(".png") or low.endswith(".jpg") or low.endswith(".jpeg"):
                items.append((os.path.join(class_dir, fn), class_to_idx[c]))

    if not items:
        raise RuntimeError("No images found in test directory.")

    _, eval_t = build_transforms(augment=False)
    dataset = EmotionFolderDataset(items, transform=eval_t)
    return dataset, class_to_idx


def save_confusion_matrix(cm, classes, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # PNG
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=classes,
        yticklabels=classes,
        cmap="Blues",
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    png_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("SAVED:", png_path)

    # JSON
    json_path = os.path.join(out_dir, "confusion_matrix.json")
    with open(json_path, "w") as f:
        json.dump(
            {"classes": classes, "matrix": cm.tolist()},
            f,
            indent=4,
        )
    print("SAVED:", json_path)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(script_dir, "checkpoints", "fer2013_b2", "best_model_b2.pth")
    data_root = os.path.join(script_dir, "..", "data", "fer2013")
    out_dir = os.path.join(script_dir, "checkpoints", "fer2013_b2")

    ckpt = load_checkpoint(checkpoint_path, device)

    # labels from checkpoint: {class_name: idx}
    labels = ckpt.get("labels")
    if labels is None:
        raise RuntimeError("Checkpoint does not contain 'labels'.")

    classes = [None] * len(labels)
    for name, idx in labels.items():
        classes[idx] = name

    dataset, _ = build_test_dataset(data_root, classes)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

    model = EmotionCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    cm = confusion_matrix(all_labels, all_preds)
    save_confusion_matrix(cm, classes, out_dir)


if __name__ == "__main__":
    main()
