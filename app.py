import os
import streamlit as st
from dotenv import load_dotenv

from orchestrator import Orchestrator
from memory.conversation_memory import ConversationMemory

load_dotenv()

st.set_page_config(
    page_title="Elsada Elafadel - Restaurant Assistant",
    page_icon="🍽️",
    layout="centered"
)

# ---- Session state setup ----
if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()

if "orchestrator" not in st.session_state:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error(
            "GROQ_API_KEY is not set. Please add it to your .env file "
            "(locally) or to Streamlit Cloud Secrets (when deployed)."
        )
        st.stop()
    st.session_state.orchestrator = Orchestrator(groq_api_key=api_key)

# ---- Header ----
st.title("🍽️ Elsada Elafadel")
st.caption("Cooked with care, served with soul. Ask me about our menu, policies, or book a table.")

# ---- Render chat history ----
for message in st.session_state.memory.get_history():
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---- Chat input ----
user_input = st.chat_input("Ask about the menu, hours, or book a table...")

if user_input:
    st.session_state.memory.add_message("user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = st.session_state.orchestrator.handle_message(
                    question=user_input,
                    history=st.session_state.memory.get_recent_history(max_messages=10)
                )
            except Exception as e:
                error_text = str(e).lower()
                if "rate" in error_text or "limit" in error_text:
                    answer = (
                        "Our assistant is a bit busy right now. "
                        "Please try again in a moment."
                    )
                else:
                    answer = (
                        "Something went wrong on my end. Please try rephrasing "
                        "your question."
                    )
        st.markdown(answer)

    st.session_state.memory.add_message("assistant", answer)


with st.sidebar:
    st.subheader("Elsada Elafadel")
    st.markdown("Branches: Tahrir, October, Shebin El Kom, Nasr City")
    st.markdown("---")
    if st.button("Clear conversation"):
        st.session_state.memory.clear()
        st.rerun()