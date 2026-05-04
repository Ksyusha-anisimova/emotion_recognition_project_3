import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from cnn_architecture import EmotionCNN
from train_fer2013_b2 import build_split_by_fraction, build_transforms, EmotionFolderDataset

# Параметры
data_root = "data/fer2013"
classes = ["happy", "sad", "neutral"]
device = "cuda" if torch.cuda.is_available() else "cpu"

# Загружаем модель
checkpoint = torch.load("model/checkpoints/best_model_b2.pth", map_location=device)
model = EmotionCNN(num_classes=len(classes)).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Формируем валидационный датасет
train_dir = f"{data_root}/train"
train_items, val_items, class_to_idx = build_split_by_fraction(
    train_dir=train_dir,
    classes=classes,
    seed=42,
    max_train_per_class=200,
    max_val_per_class=60,
    val_fraction=0.15,
)

_, eval_t = build_transforms(augment=False)
val_ds = EmotionFolderDataset(val_items, transform=eval_t)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=classes,
            yticklabels=classes,
            cmap="Blues")

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("model/checkpoints/confusion_matrix.png")
plt.show()