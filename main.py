import os
from PIL import Image
import numpy as np

# Carpeta del dataset
dataset_path = "dataset"

# Clases (nombres de las carpetas)
classes = ["nigiri_salmon", "no_nigiri"]

# Lista para datos y etiquetas
X = []  # imágenes
y = []  # etiquetas

# Recorrer cada clase
for idx, class_name in enumerate(classes):
    class_folder = os.path.join(dataset_path, class_name)
    for file in os.listdir(class_folder):
        if file.endswith((".jpg", ".png")):
            # Abrir imagen y convertirla a tamaño 64x64
            img = Image.open(os.path.join(class_folder, file)).convert("RGB").resize((64, 64))
            img_array = np.array(img)
            X.append(img_array)
            y.append(idx)

# Convertir a arrays de NumPy
X = np.array(X)
y = np.array(y)

print("Datos cargados correctamente")
print("Número de imágenes:", len(X))


from sklearn.model_selection import train_test_split

# Normalizar imágenes: poner valores entre 0 y 1
X = X / 255.0

# Dividir datos: 80% entrenamiento, 20% prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Datos preparados para entrenamiento")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Crear modelo
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)), #32 cantidad de filtros, (3,3) tamaño del filtro, activación relu funcion matematica que ayuda a aprender patrones, input_shape imagenes de 64x64 con 3 colores (RGB)
    MaxPooling2D((2,2)), #Reduce el tamaño a la mitad haciendo que la red sea más eficiente
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')  # 2 clases: nigiri_salmon y no_nigiri
])

# Compilar modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Modelo creado y compilado")

# Entrenar el modelo
history = model.fit( #Mira una imagen del X_train, predice, mira si ha acertado o no, y luego ajusta sus pesos para mejorar la próxima predicción. SI APRENDE
    X_train, y_train, #X_train = imágenes de entrenamiento, y_train = respuestas correctas para esas imágenes NO SE USAN PARA APRENDER SOLO PARA COMPARAR CON LAS PREDICCIONES
    epochs=10, #una vuelta completa a TODAS las imágenes de entrenamiento
    validation_data=(X_test, y_test) #lO MISMO QUE x_train y y_train pero con las de prueba para ver cómo va aprendiendo el modelo durante el entrenamiento
    #Comprueba si el modelo está aprendiendo de verdad o solo memorizando. Solo apunta si ha acertado o si se ha equivocado, NO APRENDE
)

print("Entrenamiento terminado")

# Evaluar el modelo
loss, accuracy = model.evaluate(X_test, y_test) #Hace un examen final y apunta el porcentaje de respuestas correctas (accuracy) y el error (loss)

print("Precisión en test:", accuracy)

def predecir_imagen(ruta_imagen):
    img = Image.open(ruta_imagen).convert("RGB").resize((64,64))
    img_array = np.array(img) / 255.0  # normalizar igual que antes
    img_array = np.expand_dims(img_array, axis=0)  # añadir dimensión extra

    prediccion = model.predict(img_array)
    clase = np.argmax(prediccion)

    if clase == 0:
        print("🍣 Es un nigiri de salmón")
    else:
        print("❌ NO es un nigiri de salmón")

# Probar con una imagen nueva
ruta = "MisFotosPrueba/nigiri_prueba.png"  # Pon aquí la ruta de tu imagen
predecir_imagen(ruta)

