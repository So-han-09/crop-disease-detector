"""
download_and_train.py
"""

import os
import sys
import time
import shutil
import json
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR    = os.path.join(BASE_DIR, 'models')
MODEL_PATH   = os.path.join(MODEL_DIR, 'plant_model.h5')
CLASSES_PATH = os.path.join(MODEL_DIR, 'class_names.json')
TFDS_DIR     = os.path.join(os.path.expanduser('~'), 'tensorflow_datasets')
IMG_SIZE     = 224
BATCH_SIZE   = 32
EPOCHS_FT    = 10
EPOCHS_FIN   = 5
MAX_RETRIES  = 10

os.makedirs(MODEL_DIR, exist_ok=True)


def clean_partial_download():
    partial = os.path.join(TFDS_DIR, 'downloads')
    if os.path.exists(partial):
        print("[INFO] Cleaning partial download cache …")
        shutil.rmtree(partial, ignore_errors=True)


def download_dataset():
    import tensorflow_datasets as tfds

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"\n[INFO] Download attempt {attempt}/{MAX_RETRIES} …")
            print("[INFO] Please keep this window open!\n")

            (ds_train, ds_val), info = tfds.load(
                'plant_village',
                split=['train[:80%]', 'train[80%:]'],
                as_supervised=True,
                with_info=True,
                shuffle_files=True,
            )

            print(f"\n✅ Dataset downloaded successfully!")
            return ds_train, ds_val, info.features['label'].names

        except KeyboardInterrupt:
            print("\n[INFO] Stopped.")
            sys.exit(0)

        except Exception as e:
            print(f"\n❌ Attempt {attempt} failed: {type(e).__name__}")
            if attempt < MAX_RETRIES:
                wait = min(attempt * 15, 120)
                clean_partial_download()
                print(f"[INFO] Retrying in {wait} seconds …")
                time.sleep(wait)
            else:
                print("\n❌ All attempts failed. Check internet and try again.")
                sys.exit(1)


def preprocess(image, label):
    import tensorflow as tf
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def augment(image, label):
    import tensorflow as tf
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def build_pipeline(ds, training=True):
    import tensorflow as tf
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.cache().shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_model(num_classes):
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import MobileNetV2

    base = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
    base.trainable = False

    inputs  = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x       = base(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dense(256, activation='relu')(x)
    x       = layers.Dropout(0.4)(x)
    x       = layers.Dense(128, activation='relu')(x)
    x       = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs), base


def train():
    import tensorflow as tf
    from tensorflow.keras import optimizers, callbacks

    print("=" * 60)
    print("  Plant Disease Detection — Training")
    print("=" * 60)

    print("\n STEP 1: Downloading Dataset …")
    ds_train_raw, ds_val_raw, class_names = download_dataset()
    num_classes = len(class_names)

    print("\n STEP 2: Preparing data …")
    train_ds = build_pipeline(ds_train_raw, training=True)
    val_ds   = build_pipeline(ds_val_raw,   training=False)

    print("\n STEP 3: Building model …")
    model, base_model = build_model(num_classes)
    model.compile(optimizer=optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    cb = [
        callbacks.EarlyStopping(patience=3, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=2, verbose=1),
        callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True, verbose=1),
    ]

    print(f"\n STEP 4: Training Phase 1 …")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FT, callbacks=cb)

    print(f"\n STEP 5: Training Phase 2 (Fine-tuning) …")
    base_model.trainable = True
    for layer in base_model.layers[:-40]:
        layer.trainable = False
    model.compile(optimizer=optimizers.Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FIN, callbacks=cb)

    print("\n STEP 6: Saving model …")
    model.save(MODEL_PATH)
    with open(CLASSES_PATH, 'w', encoding='utf-8') as f:
        json.dump(list(class_names), f, indent=2)

    print(f"\n✅ Model saved!")
    loss, acc = model.evaluate(val_ds)
    print(f"\n Accuracy: {acc*100:.2f}%")
    print("\n Run  python app.py  to start!")


if __name__ == '__main__':
    train()