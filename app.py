"""
App de Streamlit: Clasificador de radiografías de tórax COVID+/COVID-

Carga los modelos ya entrenados (pca_model.pkl y knn_model.pkl, generados
desde el notebook de entrenamiento con joblib.dump) y aplica el mismo
pipeline de preprocesamiento usado en el entrenamiento para predecir
una radiografía nueva.

Para correrla:
    pip install -r requirements.txt
    streamlit run app.py

IMPORTANTE: pca_model.pkl y knn_model.pkl deben estar en la misma carpeta
que este archivo.
"""

import numpy as np
import cv2
import joblib
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Clasificador COVID - Rayos X", page_icon="🫁")

st.title("🫁 Clasificador de radiografías de tórax")
st.markdown(
    "Sube una radiografía de tórax en escala de grises. El modelo "
    "(PCA + KNN entrenado sobre el dataset del proyecto) predecirá si "
    "corresponde a un caso **COVID+** o **COVID-**."
)


@st.cache_resource
def cargar_modelos():
    pca = joblib.load("pca_model.pkl")
    knn = joblib.load("knn_model.pkl")
    return pca, knn


try:
    pca, knn = cargar_modelos()
except FileNotFoundError:
    st.error(
        "No encontré **pca_model.pkl** o **knn_model.pkl** en esta carpeta. "
        "Genera esos archivos desde tu notebook de entrenamiento con "
        "`joblib.dump(pca, 'pca_model.pkl')` y `joblib.dump(knn, 'knn_model.pkl')`, "
        "y colócalos junto a app.py antes de correr la app."
    )
    st.stop()

TAMANO_RESIZE = (128, 128)  # debe coincidir con el resize usado en el entrenamiento


def preprocesar_imagen(imagen_pil):
    """
    Pipeline de preprocesamiento para una imagen subida por el usuario.

    El bit-depth NO se adivina a partir del contenido (eso es un
    heurístico que puede fallar). Se lee del metadato real del archivo
    (imagen_pil.mode), que distingue dos casos reales:
      - PNG de 16 bits (modos 'I', 'I;16', etc.): exportación cruda desde
        el dataset original, rango 0-16383, igual que en el entrenamiento.
      - Cualquier otro formato (JPG, PNG de 8 bits, fotos normales): rango
        estándar 0-255. JPG, de hecho, no puede ser de otra profundidad.
    """
    es_16_bits = imagen_pil.mode in ("I", "I;16", "I;16B", "I;16L")

    if es_16_bits:
        img_gris = np.array(imagen_pil, dtype=np.float32)
        divisor = 16383.0
    else:
        img_gris = np.array(imagen_pil.convert("L"), dtype=np.float32)
        divisor = 255.0

    img_norm = img_gris / divisor
    img_resized = cv2.resize(img_norm, TAMANO_RESIZE)
    vector = img_resized.reshape(1, -1)
    return vector, img_resized


archivo = st.file_uploader("Selecciona una radiografía", type=["png", "jpg", "jpeg"])

if archivo is not None:
    imagen = Image.open(archivo)
    vector, img_preprocesada = preprocesar_imagen(imagen)

    col1, col2 = st.columns(2)
    with col1:
        st.image(imagen, caption="Imagen original", use_container_width=True)
    with col2:
        st.image(
            img_preprocesada,
            caption="Imagen preprocesada (128x128)",
            use_container_width=True,
            clamp=True,
        )

    vector_pca = pca.transform(vector)
    prediccion = knn.predict(vector_pca)[0]
    probabilidades = knn.predict_proba(vector_pca)[0]

    resultado = "COVID+" if prediccion == 1 else "COVID-"
    confianza = probabilidades.max()

    st.divider()
    if resultado == "COVID+":
        st.error(f"### Resultado: {resultado}")
    else:
        st.success(f"### Resultado: {resultado}")

    st.metric("Confianza del modelo", f"{confianza:.1%}")

    st.warning(
        "⚠️ Este es un proyecto académico (KNN sobre componentes PCA de "
        "píxeles crudos). No debe usarse como herramienta de diagnóstico real."
    )

st.divider()
st.caption(
    "Modelo: PCA (50 componentes) + KNN (k=5) sobre radiografías "
    "224x224 redimensionadas a 128x128, entrenado tras eliminar imágenes "
    "corruptas del dataset original."
)
