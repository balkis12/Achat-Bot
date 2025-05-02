import streamlit as st
from Data_Generation import get_rag_chain, generate_answer_with_context

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
        color: #333;
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
}

.stTextInput input {
    font-size: 1.4rem !important;
    padding: 20px !important;
    border: none !important;
    outline: none !important;
    width: 100% !important;
    background: transparent;
}

/* Media query pour les appareils mobiles */
@media (max-width: 768px) {
    [data-testid="stChatInput"] {
        padding: 16px !important;  /* Moins de padding sur mobile */
    }

    .stTextInput input {
        font-size: 1.2rem !important;  /* Réduction de la taille du texte */
        padding: 16px !important;
    }
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

# === Sidebar
with st.sidebar:
    st.markdown("## 🤖 ChatBot IA")
    if st.button("Nouveau chat"):
        st.session_state.current_session = None
        st.session_state.first_message_sent = False

    for i, session in enumerate(st.session_state.chat_sessions):
        if st.button(session["name"], key=f"session_{i}"):
            st.session_state.current_session = i
            st.session_state.first_message_sent = True

# === Zone principale
st.markdown('<div class="chat-container">', unsafe_allow_html=True)





# Messages existants
if st.session_state.current_session is not None:
    current_session = st.session_state.chat_sessions[st.session_state.current_session]
    for message in current_session["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

st.markdown('</div>', unsafe_allow_html=True)

# === Champ d'entrée utilisateur
user_input = st.chat_input("Poser une question")



# Message de bienvenue (affiché uniquement si aucun message n'est encore envoyé)
if (
    not st.session_state.first_message_sent and
    st.session_state.current_session is None and
    not st.session_state.chat_sessions and
    user_input is None
):
    st.markdown('<div class="main-title">Comment puis-je vous aider ?</div>', unsafe_allow_html=True)




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
            result = rag_chain.invoke({"query": user_input})
            context = "\n\n".join([doc.page_content for doc in result['source_documents']])
            answer = generate_answer_with_context(llm, context, user_input)
            st.markdown(answer)
            

    current_session["messages"].append({"role": "assistant", "content": answer})

    follow_up = "Est-ce que vous avez une autre question ?"
    with st.chat_message("assistant"):
        st.markdown(follow_up)
    current_session["messages"].append({"role": "assistant", "content": follow_up})
