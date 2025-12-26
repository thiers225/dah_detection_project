import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

# Configuration de la page
st.set_page_config(
    page_title="Classification de Maladies des Plantes",
    page_icon="🌿",
    layout="centered"
)

# Constantes
MODEL_PATH = 'models/plant_disease_model.keras'
CLASS_INDICES_PATH = 'models/class_indices.json'

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

@st.cache_data
def load_class_indices():
    if not os.path.exists(CLASS_INDICES_PATH):
        return None
    with open(CLASS_INDICES_PATH, 'r') as f:
        class_indices = json.load(f)
    # Assurer que les clés sont des entiers (validation json)
    return {int(k): v for k, v in class_indices.items()}

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = image / 255.0  # Normalisation
    image = np.expand_dims(image, axis=0)  # Ajouter une dimension de batch
    return image

# Application Principale
st.title("🌿 Détection de Maladies des Plantes")
st.markdown("Téléchargez une image de feuille de plante pour détecter si elle est saine ou malade.")

model = load_model()
class_indices = load_class_indices()

if model is None or class_indices is None:
    st.error("Modèle ou indices de classe introuvables. Veuillez exécuter `notebooks/02_model_training.ipynb` d'abord pour entraîner le modèle.")
else:
    uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption='Image téléchargée', use_container_width=True)
            
            if st.button('Prédire'):
                with st.spinner('Analyse en cours...'):
                    processed_image = preprocess_image(image)
                    prediction = model.predict(processed_image)
                    
                    predicted_class_index = np.argmax(prediction, axis=1)[0]
                    confidence = np.max(prediction)
                    
                    predicted_class_name = class_indices.get(predicted_class_index, "Inconnu")
                    
                    st.success(f"**Prédiction :** {predicted_class_name}")
                    st.info(f"**Confiance :** {confidence * 100:.2f}%")
                    
        except Exception as e:
            st.error(f"Erreur lors du traitement de l'image : {e}")
