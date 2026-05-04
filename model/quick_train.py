import os
import cv2
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from cnn_architecture import EmotionCNN

# Legacy quick training for 7 classes.
# B2 version uses model/quick_train_b2.py.

# =========================
# DEVICE
# =========================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nDEVICE: {device}\n")

# =========================
# LABEL MAP (ОБЯЗАТЕЛЬНО СОВПАДАЕТ С app.py)
# =========================
emotion_to_label = {
    'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'sad': 4,
    'surprise': 5,
    'neutral': 6
}

label_to_emotion = {v: k for k, v in emotion_to_label.items()}

# =========================
# DATA LOADING
# =========================
def load_custom_data(path):
    images, labels = [], []
    printed = False
    class_count = {k: 0 for k in emotion_to_label}

    for emotion, label in emotion_to_label.items():
        folder = os.path.join(path, emotion)
        if not os.path.exists(folder):
            continue

        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (48, 48))
            img = img.astype(np.float32) / 255.0
            img = (img - 0.5) / 0.5

            if not printed:
                print("TRAIN PREPROCESS SAMPLE")
                print("file:", img_path)
                print("shape:", img.shape)
                print("dtype:", img.dtype)
                print("min:", img.min())
                print("max:", img.max())
                print("mean:", img.mean())
                print("std:", img.std())
                print("-" * 40)
                printed = True

            images.append(img)
            labels.append(label)
            class_count[emotion] += 1

    print("\nCLASS DISTRIBUTION:")
    for k, v in class_count.items():
        print(f"{k:>10}: {v}")

    X = torch.tensor(np.array(images)).unsqueeze(1)
    y = torch.tensor(labels)

    print("\nDATA SHAPES:")
    print("X:", X.shape)
    print("y:", y.shape)

    return X, y


X, y = load_custom_data('data/custom_train')

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# =========================
# MODEL
# =========================
model = EmotionCNN(num_classes=7).to(device)

# =========================
# ❄️ FREEZE CONV BLOCKS
# =========================
for name, param in model.named_parameters():
    if not name.startswith("fc"):
        param.requires_grad = False

print("\nFROZEN PARAMETERS:")
for name, param in model.named_parameters():
    if not param.requires_grad:
        print("  frozen:", name)

# =========================
# OPTIMIZER — ТОЛЬКО FC
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.001
)

# =========================
# TRAIN LOOP
# =========================
epochs = 30

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {total_loss / len(loader):.4f}")

# =========================
# TRAIN DIAGNOSTICS
# =========================
model.eval()
with torch.no_grad():
    outputs = model(X.to(device))
    probs = torch.softmax(outputs, dim=1)
    preds = probs.argmax(dim=1)

    accuracy = (preds.cpu() == y).float().mean().item()
    print(f"\nTRAIN ACCURACY: {accuracy:.3f}\n")

    print("SAMPLE PREDICTIONS (FIRST 5):")
    for i in range(min(5, len(y))):
        gt = label_to_emotion[y[i].item()]
        pr = label_to_emotion[preds[i].item()]
        p = probs[i].cpu().numpy()
        print(f"GT: {gt:>10} | PRED: {pr:>10} | PROBS: {np.round(p, 3)}")

    print("\nLOGITS STATS:")
    print("min:", outputs.min().item())
    print("max:", outputs.max().item())
    print("mean:", outputs.mean().item())
    print("std:", outputs.std().item())

# =========================
# SAVE MODEL
# =========================
os.makedirs('model/checkpoints', exist_ok=True)
torch.save(
    {'model_state_dict': model.state_dict()},
    'model/checkpoints/best_model.pth'
)

print("\n✓ MODEL SAVED: model/checkpoints/best_model.pth")
print("✓ VARIANT A TRAINING COMPLETE")
