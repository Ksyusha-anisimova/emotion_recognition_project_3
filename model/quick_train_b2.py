import os
import cv2
import torch
import numpy as np
from torch import nn, optim
from cnn_architecture import EmotionCNN

device = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", device)

# ===== B2 LABELS =====
emotion_to_label = {
    "happy": 0,
    "sad": 1,
    "neutral": 2,
}

label_to_emotion = {v: k for k, v in emotion_to_label.items()}


def load_custom_data(path):
    images, labels = [], []
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

            images.append(img)
            labels.append(label)
            class_count[emotion] += 1

    print("\nCLASS DISTRIBUTION:")
    for k, v in class_count.items():
        print(f"  {k:8s}: {v}")

    X = torch.tensor(np.array(images)).unsqueeze(1)
    y = torch.tensor(labels)

    print("\nDATA SHAPES:")
    print("X:", X.shape)
    print("y:", y.shape)

    return X.to(device), y.to(device)


# ===== LOAD DATA =====
X, y = load_custom_data("data/custom_train_b2")

# ===== MODEL =====
model = EmotionCNN(num_classes=3).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

epochs = 40

# ===== TRAIN =====
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    logits = model(X)
    loss = criterion(logits, y)

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {loss.item():.4f}")

# ===== TRAIN METRICS =====
model.eval()
with torch.no_grad():
    logits = model(X)
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)

acc = (preds == y).float().mean().item()
print("\nTRAIN ACCURACY:", round(acc, 3))

print("\nSAMPLE PREDICTIONS:")
for i in range(min(5, len(y))):
    print(
        f"GT: {label_to_emotion[y[i].item()]:7s} | "
        f"PRED: {label_to_emotion[preds[i].item()]:7s} | "
        f"PROBS: {np.round(probs[i].cpu().numpy(), 3)}"
    )

# ===== SAVE =====
os.makedirs("model/checkpoints", exist_ok=True)
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "labels": emotion_to_label,
    },
    "model/checkpoints/best_model_b2.pth",
)

print("\n✓ B2 MODEL SAVED")
