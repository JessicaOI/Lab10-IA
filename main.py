import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


def load_images(path, label, size=(128, 128)):
    images = []
    labels = []
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename))
        if img is not None:
            img_resized = cv2.resize(img, size)
            images.append(img_resized / 255.0)
            labels.append(label)
    return images, labels

cats_path = 'PetImages/Cat'
dogs_path = 'PetImages/Dog'
cats_images, cats_labels = load_images(cats_path, 0)
dogs_images, dogs_labels = load_images(dogs_path, 1)

images = cats_images + dogs_images
labels = cats_labels + dogs_labels

images, labels = shuffle(images, labels, random_state=42)

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)


# Definiendo el modelo usando TensorFlow:

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(np.array(train_images), np.array(train_labels), epochs=15, validation_split=0.1, batch_size=64)

# Evaluando el desempeño del modelo:
test_loss, test_accuracy = model.evaluate(np.array(test_images), np.array(test_labels))
print("Test accuracy:", test_accuracy)


# Mostrando la evolución de las métricas durante las diferentes épocas

def plot_metrics(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


