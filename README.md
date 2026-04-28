# CNN-Framework-with-Infrared-and-RMS-Features-for-Detecting-Arc-EDM-
https://colab.research.google.com/gist/aialisam250/c5c061ce71f583d1994bcfca1360b5b9/cnn-framework-with-infrared-and-rms-features-for-detecting-arc-edm.ipynb


import zipfile
import os

zip_filename = "votre_fichier.zip"

# Unzip
if os.path.exists(zip_filename):
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall('.')
    print(" Extraction complete.")
else:
    print(" Error: Zip file not found. Make sure you uploaded it.")

# Check the extracted folders
print("Folders found:", os.listdir('.'))
# see ['train', 'test', ...] in this list

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

# --- 1. UPDATED PARAMETERS FOR YOUR ZIP STRUCTURE ---
img_width, img_height = 224, 224

# CHANGE: Removed 'images/' because your folders are at the root
train_data_dir = 'train'
validation_data_dir = 'test'

nb_train_samples = 400  # Adjust if you have more/less images
nb_validation_samples = 100
epochs = 15
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# --- 2. DATA GENERATION ---
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

# Verify that Python finds the images
print("--- Loading Training Data ---")
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

print("--- Loading Test Data ---")
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

# --- 3. ARCHITECTURE  ---
model = Sequential()
model.add(Conv2D(8, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(24))
model.add(Activation('relu'))
model.add(Dense(2))  # 2 neurons because class_mode='categorical'
model.add(Activation('sigmoid'))

print(model.summary())

# --- 4. TRAINING ---
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Start Training
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# --- 5. VISUALIZATION ---
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.title('Accuracy')
plt.legend()
plt.show()
