import os
import re

import base64
from PIL import Image
import streamlit as st
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from Data_Generation import get_rag_chain, generate_answer_with_context
from prediction_model import predict_quantity  # Import de la fonction de prédiction

# Initialisation du modèle
rag_chain, llm = get_rag_chain()

# Configuration de l'application
st.set_page_config(page_title="ChatBot IA - Achats", page_icon="🤖", layout="centered")

# === Style CSS
st.markdown("""
<style>
    .stApp {
        background: #ffffff;
        font-family: Arial, sans-serif;
    }

    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #0056A03;
        margin-top: 40px;
        margin-bottom: 10px;
    }

    .chat-container {
        max-width: 1000px;
        margin: 0 auto;
        padding: 20px;
    }

    [data-testid="stChatMessage"] {
        max-width: 1000px;
        margin: 20px 0 !important;
    }

    [data-testid="stChatInput"] {
        width: 100% !important;
        max-width: 100% !important;
        margin: 10px 0 30px 0 !important;
        position: relative;
        top: -30px;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.2);
        border-radius: 16px;
        background: white;
        padding: 20px;
        border-bottom: 2px solid #E3F0FA;
    }

    .stTextInput input {
        font-size: 1.4rem !important;
        padding: 20px !important;
        border: none !important;
        outline: none !important;
        width: 100% !important;
        background: transparent;
    }

    @media (max-width: 768px) {
        [data-testid="stChatInput"] {
            padding: 16px !important;
        }

        .stTextInput input {
            font-size: 1.2rem !important;
            padding: 16px !important;
        }
    }

    div[data-testid="stSidebar"] {
        background-color: #E3F0FA !important;
        border-left: 5px solid #E3F0FA !important;
    }

</style>
""", unsafe_allow_html=True)

# === Initialisation de session
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = []
if "current_session" not in st.session_state:
    st.session_state.current_session = None
if "first_message_sent" not in st.session_state:
    st.session_state.first_message_sent = False


# === Charger le logo en base64
logo_path = "images/LOGO-MENU.png"  
with open(logo_path, "rb") as image_file:
    logo_base64 = base64.b64encode(image_file.read()).decode("utf-8")

# Sidebar
with st.sidebar:
  # Logo + titre centrés
    st.markdown(f"""
    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
        <img src="data:image/png;base64,{logo_base64}" alt="Logo Délice" style="height: 65px; margin-right: 10px;">
        <span style="font-size: 35px; font-weight: bold;">
            <span style="color: #0056A0;">Achat</span>
            <span style="color: #E30613;">Bot</span>
        </span>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Nouveau chat"):
        st.session_state.current_session = None
        st.session_state.first_message_sent = False

    for i, session in enumerate(st.session_state.chat_sessions):
        if st.button(session["name"], key=f"session_{i}"):
            st.session_state.current_session = i
            st.session_state.first_message_sent = True

    st.markdown("---")
    st.markdown("""
    <style>
    div[data-testid="stSidebar"] {
        background-color: #E3F0FA !important;
    }
    div[data-testid="stFileUploader"] > div > div > div > div > p {
        display: none !important;
    }
    div[data-testid="stFileUploader"] > div > div > div > div {
        padding: 0 !important;
        margin: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

      # Remplacer la partie "Browse files" par :
    uploaded_files = st.file_uploader(
        "📤 Uploader des documents",
        accept_multiple_files=True,
        type=["pdf", "txt"],
        key="file_uploader"
    )
# === Traitement des fichiers
if uploaded_files:
    user_docs = []
    for uploaded_file in uploaded_files:
        file_path = f"./temp_files/{uploaded_file.name}"
        os.makedirs("temp_files", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif uploaded_file.name.endswith(".txt"):
            loader = TextLoader(file_path)
        else:
            st.warning(f"❌ Format non supporté : {uploaded_file.name}")
            continue

        loaded_docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = splitter.split_documents(loaded_docs)
        user_docs.extend(split_docs)

    # Sauvegarde dans la session
    st.session_state.user_docs = user_docs

    


# === Zone principale
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Affichage des messages
if st.session_state.current_session is not None:
    current_session = st.session_state.chat_sessions[st.session_state.current_session]
    for message in current_session["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

st.markdown('</div>', unsafe_allow_html=True)

# === Champ de saisie utilisateur
user_input = st.chat_input("Poser une question")

# Message de bienvenue
if (
    not st.session_state.first_message_sent and
    st.session_state.current_session is None and
    not st.session_state.chat_sessions and
    user_input is None
):
    st.markdown('<div class="main-title" style="color: #0056A0;">Comment puis-je vous aider ?</div>', unsafe_allow_html=True)

# === Réponse du chatbot
if user_input:
    st.session_state.first_message_sent = True

    if st.session_state.current_session is None:
        new_session = {
            "name": user_input[:30] + "..." if len(user_input) > 30 else user_input,
            "messages": []
        }
        st.session_state.chat_sessions.append(new_session)
        st.session_state.current_session = len(st.session_state.chat_sessions) - 1

    current_session = st.session_state.chat_sessions[st.session_state.current_session]
    current_session["messages"].append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Génération de la réponse..."):
            try:
                if "quantité" in user_input.lower():
                    predicted_quantity = predict_quantity(user_input)
                    
                    # Extraction de l'unité depuis la question (par exemple : (PI), (litres))
                    match_unite = re.search(r'\(([^)]*)\)', user_input)
                    if match_unite:
                        unite_extraite = match_unite.group(1)
                    else:
                        unite_extraite = "unités"  # Par défaut si aucune unité trouvée

                    response_quantite = f"La quantité prévue est : {predicted_quantity} {unite_extraite}."
                    # Enrichissement du contexte avec la réponse quantité et quelques détails sur la question
                    context = f"Question: {user_input}\nInformations extraites: quantité prévue = {predicted_quantity} {unite_extraite}.\nMerci de répondre en tenant compte de ces données."
                    # Génération LLM réponse
                    response_llm = generate_answer_with_context(llm,context, user_input)
                    # Fusionner les deux réponses
                    response = response_quantite + "\n\n" + response_llm
                elif "document" in user_input.lower():
                   if 'user_docs' in locals() and user_docs:
                       context = "\n\n".join([doc.page_content for doc in user_docs])
                       response = generate_answer_with_context(llm, context, user_input)
                   else:
                       response = "❌ Aucun document n'a été chargé. Veuillez uploader un fichier d'abord."

                else:
                    result = rag_chain.invoke({"query": user_input})
                    context = "\n\n".join([doc.page_content for doc in result['source_documents']])
                    response = generate_answer_with_context(llm, context, user_input)
                st.markdown(response)
            except Exception as e:
                st.error(f"❌ Erreur : {e}")

    current_session["messages"].append({"role": "assistant", "content": response})

    follow_up = "Est-ce que vous avez une autre question ?"
    with st.chat_message("assistant"):
        st.markdown(follow_up)
    current_session["messages"].append({"role": "assistant", "content": follow_up})





