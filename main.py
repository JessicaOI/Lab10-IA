import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def load_images_and_labels(paths, labels, size=(128, 128)):
    filepaths = [(os.path.join(path, filename), label) for path, label in zip(paths, labels) for filename in os.listdir(path)]
    images = []
    image_labels = []

    for (filepath, label) in filepaths:
        try:
            img = tf.io.read_file(filepath)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, size)
            img = img / 255.0
            images.append(img)
            image_labels.append(label)
        except Exception as e:
            print(f"Error reading file {filepath}: {e}")
            continue

    return tf.data.Dataset.from_tensor_slices((images, image_labels))



cats_path = 'PetImages/Cat'
dogs_path = 'PetImages/Dog'
paths = [cats_path, dogs_path]
labels = [0, 1]

dataset = load_images_and_labels(paths, labels)

dataset = dataset.shuffle(10000, seed=42)

total_size = len(dataset)

train_size = int(0.8 * total_size)
train_ds = dataset.take(train_size)
test_ds = dataset.skip(train_size)

train_ds = train_ds.batch(64).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.batch(64).prefetch(tf.data.AUTOTUNE)

# Definiendo el modelo usando TensorFlow
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

# Agrega detención temprana
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# Entrena el modelo
history = model.fit(train_ds, epochs=15, validation_data=test_ds, callbacks=[early_stopping])

# Evaluando el desempeño del modelo
test_loss, test_accuracy = model.evaluate(test_ds)
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

plot_metrics(history)

# Guarda el modelo entrenado
model.save("cat_dog_classifier.h5")
