Para comenzar el github no me dejo subir el .h5 que contiene el ya entrenado porque pesaba mas de 100mb pero lo subi en el zip que subi en canvas para que no tengan que volver a correr todo el entrenamiento del agente que si me tardo un buen rato.

Pero si quisieran ejecutar el entrenamiento que seria el main.py se deben de asegurar que tiene descargada el dataset de imagenes de perros y gatos (https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset?resource=download)

Lo que descarguen lo extraen y en la misma ubicacion que estan estos archivos colocar la carpeta llamada "PetImages"

Para entrenar el agente IA para que reconozca una imagen de un perro o un gato se ejecuta el main.py
el cual al final de evaluar por Epoch las imagenes de perros y gatos me devolvio lo siguiente:
Test accuracy: 0.8728304505348206
Tambien una grafica que se puede ver en la grafica.png

Luego para usar el agente IA y ver si reconoce una imagen de un gato o un perro se hace ejecuanto el archivo predictor.py
se le pide el nombre de la imagen, procede a evaluarla, dice que piensa que es y la probabilidad de acierto