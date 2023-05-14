import os
import tensorflow as tf

def load_and_preprocess_image(filepath, size=(128, 128)):
    try:
        img = tf.io.read_file(filepath)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, size)
        img = img / 255.0
        img = tf.expand_dims(img, axis=0)
        return img
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None

def predict_image(model, filepath):
    img = load_and_preprocess_image(filepath)
    if img is not None:
        prediction = model.predict(img)
        probability = prediction[0][0]
        if probability > 0.5:
            print(f"{filepath} es un perro con una probabilidad del {probability * 100:.2f}%.")
        else:
            print(f"{filepath} es un gato con una probabilidad del {(1 - probability) * 100:.2f}%.")
    else:
        print(f"Error al procesar la imagen {filepath}")


# Carga el modelo previamente entrenado
model = tf.keras.models.load_model("cat_dog_classifier.h5")

# Pide al usuario el nombre de un archivo .jpg
filename = input("Ingresa el nombre de un archivo y la extension .jpg, formato: nombre_archivo.jpg ")

# Realiza la predicción
predict_image(model, filename)
