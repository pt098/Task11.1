import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16

# Paths to your datasets
train_path = r'C:\Arduino and raspberry Pi projects\project\data\resized_normal'
tsunami_path = r'C:\Arduino and raspberry Pi projects\project\data\resized_tsunami'

# Create combined data directory
combined_train_path = r'C:\Arduino and raspberry Pi projects\project\data\combined_train'
os.makedirs(combined_train_path, exist_ok=True)
os.makedirs(os.path.join(combined_train_path, 'normal'), exist_ok=True)
os.makedirs(os.path.join(combined_train_path, 'tsunami'), exist_ok=True)

# Copy images to combined directory (instead of linking)
for filename in os.listdir(train_path):
    src_path = os.path.join(train_path, filename)
    dest_path = os.path.join(combined_train_path, 'normal', filename)
    if not os.path.exists(dest_path):  # Check if the file already exists
        shutil.copy(src_path, dest_path)

for filename in os.listdir(tsunami_path):
    src_path = os.path.join(tsunami_path, filename)
    dest_path = os.path.join(combined_train_path, 'tsunami', filename)
    if not os.path.exists(dest_path):  # Check if the file already exists
        shutil.copy(src_path, dest_path)

# Data preprocessing with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Set validation split
)

train_gen = train_datagen.flow_from_directory(
    combined_train_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_gen = train_datagen.flow_from_directory(
    combined_train_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Transfer learning with VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=0.00001)
checkpoint = ModelCheckpoint('model_checkpoint.keras', save_best_only=True, monitor='val_loss', save_freq='epoch')

# Train the model
history = model.fit(
    train_gen,
    epochs=20,
    validation_data=val_gen,
    callbacks=[early_stopping, lr_reduction, checkpoint]
)

# Unfreeze some layers for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:5]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

fine_tune_history = model.fit(
    train_gen,
    epochs=20,
    validation_data=val_gen,
    callbacks=[early_stopping, lr_reduction, checkpoint]
)

# Plotting the training history
plt.figure(figsize=(12, 4))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(fine_tune_history.history['accuracy'], label='Fine-tuning Accuracy')
plt.plot(fine_tune_history.history['val_accuracy'], label='Fine-tuning Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(fine_tune_history.history['loss'], label='Fine-tuning Loss')
plt.plot(fine_tune_history.history['val_loss'], label='Fine-tuning Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.show()

# Save the final model
model.save('tsunami_detection_model.keras')
