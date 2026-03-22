"""
train_model.py
==============
Trains a MobileNetV2-based CNN on the PlantVillage dataset and saves
the model as  models/plant_model.h5

Usage:
    python train_model.py

The script will:
  1. Auto-download PlantVillage via tensorflow-datasets (or use local data).
  2. Apply data augmentation.
  3. Fine-tune MobileNetV2.
  4. Save the model + class-name list.
"""

import os
import json
import numpy as np

# ── TensorFlow ──────────────────────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2

# ── Config ──────────────────────────────────────────────────────────────────
IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS_FT   = 10   # feature-extraction epochs
EPOCHS_FIN  = 5    # fine-tuning epochs
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, 'models')
MODEL_PATH  = os.path.join(MODEL_DIR, 'plant_model.h5')
CLASSES_PATH = os.path.join(MODEL_DIR, 'class_names.json')

os.makedirs(MODEL_DIR, exist_ok=True)

# ── PlantVillage class list (38 classes) ─────────────────────────────────────
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]
NUM_CLASSES = len(CLASS_NAMES)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_from_tfds():
    """Load PlantVillage via tensorflow-datasets (auto-downloads ~800 MB)."""
    print("[INFO] Loading PlantVillage dataset via tensorflow-datasets …")
    import tensorflow_datasets as tfds

    (ds_train_raw, ds_val_raw), info = tfds.load(
        'plant_village',
        split=['train[:80%]', 'train[80%:]'],
        as_supervised=True,
        with_info=True,
        shuffle_files=True,
    )

    tfds_class_names = info.features['label'].names
    print(f"[INFO] Dataset loaded. Classes: {info.features['label'].num_classes}")
    return ds_train_raw, ds_val_raw, tfds_class_names


def load_from_directory(data_dir):
    """Load from a local directory structured as ImageFolder."""
    print(f"[INFO] Loading dataset from directory: {data_dir}")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset='training',
        seed=42,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset='validation',
        seed=42,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
    )
    return train_ds, val_ds, train_ds.class_names


def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def build_pipeline(ds, training=True, tfds_mode=True):
    if tfds_mode:
        ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache().shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE) if tfds_mode else \
         ds.cache().prefetch(tf.data.AUTOTUNE)
    return ds


# ── Model building ────────────────────────────────────────────────────────────

def build_model(num_classes):
    base = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
    )
    base.trainable = False  # freeze for feature extraction phase

    inputs  = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x       = base(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dense(256, activation='relu')(x)
    x       = layers.Dropout(0.4)(x)
    x       = layers.Dense(128, activation='relu')(x)
    x       = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model, base


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    # ── Decide data source ────────────────────────────────────────────────────
    local_data = os.path.join(BASE_DIR, 'data', 'PlantVillage')
    tfds_mode  = not os.path.isdir(local_data)

    if tfds_mode:
        ds_train_raw, ds_val_raw, tfds_class_names = load_from_tfds()
        train_ds = build_pipeline(ds_train_raw, training=True,  tfds_mode=True)
        val_ds   = build_pipeline(ds_val_raw,   training=False, tfds_mode=True)
        # Use the TFDS class ordering; remap to our CLASS_NAMES order if needed
        num_classes = len(tfds_class_names)
    else:
        train_ds, val_ds, dir_class_names = load_from_directory(local_data)
        tfds_class_names = dir_class_names
        num_classes = len(dir_class_names)

    print(f"[INFO] Number of classes: {num_classes}")

    # ── Build model ───────────────────────────────────────────────────────────
    model, base_model = build_model(num_classes)
    model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    model.summary()

    cb = [
        callbacks.EarlyStopping(patience=3, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=2, verbose=1),
        callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True, verbose=1),
    ]

    # Phase 1 – feature extraction
    print("\n[PHASE 1] Feature extraction …")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FT, callbacks=cb)

    # Phase 2 – fine-tune top 40 layers of MobileNetV2
    print("\n[PHASE 2] Fine-tuning …")
    base_model.trainable = True
    for layer in base_model.layers[:-40]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FIN, callbacks=cb)

    # ── Save artefacts ────────────────────────────────────────────────────────
    model.save(MODEL_PATH)
    print(f"[INFO] Model saved → {MODEL_PATH}")

    # Save class names (use the ones that came from the actual dataset)
    used_classes = tfds_class_names if tfds_mode else dir_class_names
    with open(CLASSES_PATH, 'w', encoding='utf-8') as f:
        json.dump(list(used_classes), f, indent=2)
    print(f"[INFO] Class names saved → {CLASSES_PATH}")

    # Evaluate
    print("\n[INFO] Final evaluation on validation set:")
    loss, acc = model.evaluate(val_ds)
    print(f"  Validation accuracy: {acc*100:.2f}%")


if __name__ == '__main__':
    train()
