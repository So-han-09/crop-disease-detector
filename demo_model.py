"""
demo_model.py
=============
Creates a LIGHTWEIGHT DEMO model (no real training) so you can test the
web application immediately without waiting for full model training.

The demo model is an untrained MobileNetV2 — predictions will be random,
but it lets you verify the full pipeline works end-to-end.

Usage:
    python demo_model.py

After running, start the server:
    python app.py
"""

import os
import json
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR    = os.path.join(BASE_DIR, 'models')
MODEL_PATH   = os.path.join(MODEL_DIR, 'plant_model.h5')
CLASSES_PATH = os.path.join(MODEL_DIR, 'class_names.json')

os.makedirs(MODEL_DIR, exist_ok=True)

CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot",
    "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight",
    "Potato___healthy", "Raspberry___healthy",
    "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight",
    "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus", "Tomato___healthy",
]

NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE    = 224


def build_demo_model():
    base = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights=None,           # no ImageNet weights — purely random
    )
    inputs  = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x       = base(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dense(256, activation='relu')(x)
    x       = layers.Dropout(0.4)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    return models.Model(inputs, outputs)


if __name__ == '__main__':
    print("[INFO] Building demo model (random weights — for UI testing only) …")
    model = build_demo_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.save(MODEL_PATH)
    print(f"[INFO] Demo model saved → {MODEL_PATH}")

    with open(CLASSES_PATH, 'w', encoding='utf-8') as f:
        json.dump(CLASS_NAMES, f, indent=2)
    print(f"[INFO] Class names saved → {CLASSES_PATH}")

    # Quick sanity check
    dummy = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
    preds = model.predict(dummy, verbose=0)
    top   = int(np.argmax(preds[0]))
    print(f"[INFO] Sanity check — predicted class: {CLASS_NAMES[top]}")
    print("\n✅  Demo model ready. Run  python app.py  to start the server.")
    print("   ⚠️  For accurate predictions, run  python train_model.py  instead.")
