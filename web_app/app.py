"""
Flask веб-приложение для распознавания эмоций (5 классов)
Классы: happy / sad / neutral / angry / surprise
"""

from flask import Flask, render_template, request, jsonify
import torch
import cv2
import numpy as np
import base64
import os
import sys

# ─────────────────────────────────────────────────────────────
# Пути и импорты модели
# ─────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "model"))
CHECKPOINT_PATH = os.environ.get(
    "EMOTION_CHECKPOINT_PATH",
    os.path.join(MODEL_DIR, "checkpoints", "best_model_5c.pth"),
)

sys.path.append(MODEL_DIR)

from cnn_architecture import EmotionCNN

# ─────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

device = "cuda" if torch.cuda.is_available() else "cpu"
model = None

# ─────────────────────────────────────────────────────────────
# Эмоции (5 классов)
# Порядок ДОЛЖЕН совпадать с обучением
# 0: happy, 1: sad, 2: neutral, 3: angry, 4: surprise
# ─────────────────────────────────────────────────────────────

EMOTION_LABELS = {
    0: {"ru": "Счастье", "en": "Happy", "emoji": "😊", "color": "#FFD93D"},
    1: {"ru": "Грусть", "en": "Sad", "emoji": "😢", "color": "#6BCB77"},
    2: {"ru": "Нейтральное", "en": "Neutral", "emoji": "😐", "color": "#C7CEEA"},
    3: {"ru": "Злость", "en": "Angry", "emoji": "😠", "color": "#FF6B6B"},
    4: {"ru": "Удивление", "en": "Surprise", "emoji": "😲", "color": "#4D96FF"},
}

# ─────────────────────────────────────────────────────────────
# Загрузка модели (5 классов)
# ─────────────────────────────────────────────────────────────

def load_model():
    global model

    print("Загрузка модели из:", CHECKPOINT_PATH)
    print("Файл существует:", os.path.exists(CHECKPOINT_PATH))

    model = EmotionCNN(num_classes=5).to(device)

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"❌ Чекпоинт 5-классовой модели не найден: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("✓ 5-классовая модель успешно загружена")

# ─────────────────────────────────────────────────────────────
# Face detection
# ─────────────────────────────────────────────────────────────

def detect_face(image):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    return faces

# ─────────────────────────────────────────────────────────────
# Preprocess (СТРОГО как в обучении B2)
# ─────────────────────────────────────────────────────────────

def preprocess_image(image):
    faces = detect_face(image)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_img = image[y:y+h, x:x+w]
        face_roi = (int(x), int(y), int(w), int(h))
    else:
        face_img = image
        face_roi = None

    if len(face_img.shape) == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

    face_img = cv2.resize(face_img, (48, 48))
    face_img = face_img.astype(np.float32) / 255.0
    face_img = (face_img - 0.5) / 0.5

    # диагностический лог (можно оставить для диплома)
    print("INFER PREPROCESS SAMPLE")
    print("shape:", face_img.shape)
    print("min:", face_img.min(), "max:", face_img.max())
    print("mean:", face_img.mean(), "std:", face_img.std())
    print("-" * 40)

    tensor = torch.from_numpy(face_img).unsqueeze(0).unsqueeze(0).to(device)
    return tensor, face_roi

# ─────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────

def predict_emotion(image):
    tensor, face_roi = preprocess_image(image)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    emotion_id = int(np.argmax(probs))
    confidence = float(probs[emotion_id])

    result = {
        "emotion_id": emotion_id,
        "emotion_ru": EMOTION_LABELS[emotion_id]["ru"],
        "emotion_en": EMOTION_LABELS[emotion_id]["en"],
        "emoji": EMOTION_LABELS[emotion_id]["emoji"],
        "color": EMOTION_LABELS[emotion_id]["color"],
        "confidence": confidence,
        "all_probabilities": {
            EMOTION_LABELS[i]["ru"]: float(probs[i])
            for i in range(len(probs))
        },
        "face_detected": face_roi is not None,
        "face_roi": face_roi,
    }

    return result

# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Файл не передан"}), 400

        file = request.files["file"]
        image_bytes = file.read()

        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Некорректное изображение"}), 400

        result = predict_emotion(image)

        if result["face_roi"]:
            x, y, w, h = result["face_roi"]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        _, buffer = cv2.imencode(".jpg", image)
        img_base64 = base64.b64encode(buffer).decode("utf-8")
        result["processed_image"] = f"data:image/jpeg;base64,{img_base64}"

        if result["face_roi"] is not None:
            result["face_roi"] = [int(v) for v in result["face_roi"]]

        return jsonify(result)

    except Exception as e:
        print("Ошибка /predict:", e)
        return jsonify({"error": "Ошибка при анализе изображения"}), 500

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_model()

    print("=" * 70)
    print("ЗАПУСК ВЕБ-ПРИЛОЖЕНИЯ (5 CLASSES)")
    print("Устройство:", device)
    print("http://localhost:5000")
    print("=" * 70)

    app.run(debug=True, host="0.0.0.0", port=5000)
