import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# Configurar la página
st.set_page_config(
    page_title="Detector de Nigiri de Salmón",
    page_icon="🍣",
    layout="centered"
)

st.title("🍣 Detector de Nigiri de Salmón")
st.write("Sube una foto y te diré si es un nigiri de salmón o no")

# Cargar el modelo (con cache para que no se recargue cada vez)
@st.cache_resource
def cargar_modelo():
    if not os.path.exists("modelo_nigiri.h5"):
        st.error("❌ El modelo no existe. Por favor, ejecuta main.py primero para entrenar el modelo.")
        st.stop()
    return tf.keras.models.load_model("modelo_nigiri.h5")

model = cargar_modelo()

# Crear dos columnas para la interfaz
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Subir imagen")
    archivo_cargado = st.file_uploader(
        "Elige una imagen",
        type=["jpg", "png", "jpeg"],
        help="Sube una foto en formato JPG o PNG"
    )

with col2:
    st.subheader("Vista previa")
    if archivo_cargado is not None:
        imagen = Image.open(archivo_cargado)
        st.image(imagen, use_column_width=True)

# Realizar predicción si hay imagen cargada
if archivo_cargado is not None:
    st.divider()
    
    if st.button("🔍 Analizar imagen", type="primary", use_container_width=True):
        with st.spinner("Analizando imagen..."):
            # Preparar la imagen
            img = Image.open(archivo_cargado).convert("RGB").resize((64, 64))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Hacer predicción
            prediccion = model.predict(img_array, verbose=0)
            confianza = np.max(prediccion) * 100
            clase = np.argmax(prediccion)
            
            # Mostrar resultado
            st.divider()
            st.subheader("Resultado del análisis")
            
            if clase == 0:
                st.success(f"✅ **Es un nigiri de salmón**")
                st.info(f"Confianza: {confianza:.1f}%")
            else:
                st.warning(f"❌ **NO es un nigiri de salmón**")
                st.info(f"Confianza: {confianza:.1f}%")

st.divider()
st.caption("Modelo entrenado con CNN (Convolutional Neural Network)")
