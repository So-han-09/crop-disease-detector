import os, json
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, 'data', 'PlantVillage')
MODEL_DIR    = os.path.join(BASE_DIR, 'models')
MODEL_PATH   = os.path.join(MODEL_DIR, 'plant_model.h5')
CLASSES_PATH = os.path.join(MODEL_DIR, 'class_names.json')
IMG_SIZE     = 224
BATCH_SIZE   = 16

os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading dataset...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset='training',
    seed=42, image_size=(IMG_SIZE,IMG_SIZE), batch_size=BATCH_SIZE)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset='validation',
    seed=42, image_size=(IMG_SIZE,IMG_SIZE), batch_size=BATCH_SIZE)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Classes found: {num_classes}")

augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.2),
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.2),
])

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(lambda x,y: (augment(x,training=True)/255.0, y), num_parallel_calls=AUTOTUNE).cache().shuffle(2000).prefetch(AUTOTUNE)
val_ds   = val_ds.map(lambda x,y: (x/255.0, y), num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)

base = MobileNetV2(input_shape=(IMG_SIZE,IMG_SIZE,3), include_top=False, weights='imagenet')
base.trainable = False

inputs  = layers.Input(shape=(IMG_SIZE,IMG_SIZE,3))
x       = base(inputs, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.BatchNormalization()(x)
x       = layers.Dense(512, activation='relu')(x)
x       = layers.Dropout(0.5)(x)
x       = layers.Dense(256, activation='relu')(x)
x       = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)
model   = models.Model(inputs, outputs)

model.compile(optimizer=optimizers.Adam(1e-3),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

cb = [
    callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(factor=0.3, patience=3, verbose=1),
    callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True, verbose=1),
]

print("\nPhase 1 - Feature Extraction (20 epochs)...")
model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=cb)

print("\nPhase 2 - Fine Tuning...")
base.trainable = True
for layer in base.layers[:-60]:
    layer.trainable = False

model.compile(optimizer=optimizers.Adam(1e-4),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=cb)

model.save(MODEL_PATH)
with open(CLASSES_PATH, 'w') as f:
    json.dump(class_names, f, indent=2)

print("\nFinal evaluation:")
loss, acc = model.evaluate(val_ds)
print(f"Validation Accuracy: {acc*100:.2f}%")
print("\nDone! Run python app.py to start!")