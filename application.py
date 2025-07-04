import os
from dotenv import load_dotenv
import streamlit as st
import requests
from transformers import pipeline

# Charge le fichier .env
load_dotenv()

CLIPDROP_API_KEY = os.getenv("CLIPDROP_API_KEY")

if not CLIPDROP_API_KEY:
    st.error("La clé API n'a pas été trouvée. Vérifiez votre fichier .env.")

st.set_page_config(page_title="Synthétiseur de rêves", page_icon="🌙")

def generate_image(prompt: str) -> bytes | None:
    url = "https://clipdrop-api.co/text-to-image/v1"
    headers = {
        "x-api-key": CLIPDROP_API_KEY,
        "Content-Type": "application/json"
    }
    data = {"prompt": prompt}

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.content
    else:
        st.error(f"Erreur {response.status_code} : {response.text}")
        return None

@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True
    )

emotion_model = load_emotion_model()

def detect_emotion(text: str):
    results = emotion_model(text)[0]
    top = max(results, key=lambda x: x["score"])
    return top["label"], top["score"], results

st.title("🌌 Synthétiseur de rêves")
st.write("Racontez un rêve et voyez-le prendre vie sous forme d'image.")

mode = st.radio("Méthode d’entrée :", ["Écrire un rêve", "Uploader un fichier audio"])

texte_reve = ""
if mode == "Écrire un rêve":
    texte_reve = st.text_area("Décrivez votre rêve :", height=150)
else:
    audio = st.file_uploader("Téléversez un fichier audio (.mp3 ou .wav)", type=["mp3", "wav"])
    if audio:
        st.success("Fichier audio reçu. (Transcription simulée pour l’instant)")
        texte_reve = "Je rêve que Messi joue au football avec des extraterrestres dans l’espace."

if texte_reve.strip():
    st.subheader("Texte du rêve")
    st.write(texte_reve)

    if st.button("Générer l’image"):
        with st.spinner("Création de l’image..."):
            img_bytes = generate_image(texte_reve)
        if img_bytes:
            st.image(img_bytes, caption="Image générée", use_column_width=True)

    if st.button("Analyser l’émotion"):
        emotion, score, details = detect_emotion(texte_reve)
        st.markdown(f"**Émotion dominante : `{emotion}`** (score : `{score:.2f}`)")
        st.markdown("Détails des scores :")
        for e in details:
            st.write(f"- {e['label']} : {e['score']:.2f}")

    if "historique" not in st.session_state:
        st.session_state["historique"] = []

    if st.button("Sauvegarder ce rêve"):
        st.session_state["historique"].append(texte_reve)
        st.success("Rêve sauvegardé.")

    if st.session_state["historique"]:
        st.subheader("Historique")
        for i, r in enumerate(reversed(st.session_state["historique"]), 1):
            st.markdown(f"{i}. {r}")
else:
    st.info("Veuillez écrire ou uploader un rêve pour commencer.")
