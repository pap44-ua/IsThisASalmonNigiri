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
    if not os.path.exists(class_folder):
        print(f"⚠️ Carpeta {class_folder} no encontrada")
        continue
    
    num_imagenes_clase = 0
    for file in os.listdir(class_folder):
        if file.endswith((".jpg", ".png", ".jpeg")):
            try:
                # Abrir imagen y convertirla a tamaño 128x128 (más resolución)
                img = Image.open(os.path.join(class_folder, file)).convert("RGB").resize((128, 128))
                img_array = np.array(img)
                X.append(img_array)
                y.append(idx)
                num_imagenes_clase += 1
            except Exception as e:
                print(f"Error cargando {file}: {e}")
    
    print(f"✓ {class_name}: {num_imagenes_clase} imágenes")

# Convertir a arrays de NumPy
X = np.array(X)
y = np.array(y)

print("\n✓ Datos cargados correctamente")
print(f"  Total de imágenes: {len(X)}")
print(f"  Nigiri salmón: {np.sum(y == 0)}")
print(f"  No nigiri: {np.sum(y == 1)}")


from sklearn.model_selection import train_test_split

# Normalizar imágenes: poner valores entre 0 y 1
X = X / 255.0

# Dividir datos: 80% entrenamiento, 20% prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n✓ Datos preparados para entrenamiento")
print(f"  Entrenamiento: {len(X_train)} imágenes")
print(f"  Prueba: {len(X_test)} imágenes")

# Data Augmentation: generar variaciones de las imágenes de entrenamiento
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,  # Rotar imágenes 0-20 grados
    width_shift_range=0.2,  # Desplazar horizontalmente 20%
    height_shift_range=0.2,  # Desplazar verticalmente 20%
    horizontal_flip=True,  # Voltear horizontalmente
    zoom_range=0.2,  # Zoom 0.8x a 1.2x
    brightness_range=[0.8, 1.2]  # Variar brillo
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Crear modelo más potente
model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(256, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),  # Evita overfitting
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')  # 2 clases: nigiri_salmon y no_nigiri
])

# Compilar modelo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n✓ Modelo creado y compilado")

# Entrenar el modelo con Data Augmentation
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),  # Data augmentation en tiempo real
    epochs=50,  # Más épocas para mejor aprendizaje
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

print("\n✓ Entrenamiento terminado")

# Evaluar el modelo
loss, accuracy = model.evaluate(X_test, y_test)

print(f"\n✓ Precisión en test: {accuracy*100:.2f}%")
print(f"  Error (loss): {loss:.4f}")

print("Precisión en test:", accuracy)

# Guardar el modelo entrenado
model.save("modelo_nigiri.h5")
print("Modelo guardado como 'modelo_nigiri.h5'")

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
ruta = "MisFotosPrueba/prueba.png"  # Pon aquí la ruta de tu imagen
if os.path.exists(ruta):
    predecir_imagen(ruta)

