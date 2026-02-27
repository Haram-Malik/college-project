import streamlit as st
from jarvis.config import settings
from jarvis.ollama_client import OllamaClient
from jarvis.router import Router
from jarvis.tts import Speaker

st.set_page_config(page_title="Offline Jarvis (Ollama)", page_icon="🤖", layout="centered")

llm = OllamaClient(base_url=settings.OLLAMA_BASE_URL, model=settings.OLLAMA_MODEL)
speaker = Speaker(enabled=False)  # UI: keep silent by default
router = Router(llm=llm, speaker=speaker)

st.title("🤖 Offline Jarvis (Ollama)")
st.caption("Offline LLM + automation + document assistant (no cloud).")

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.subheader("Settings")
    st.write(f"Model: `{settings.OLLAMA_MODEL}`")
    st.write(f"Ollama URL: `{settings.OLLAMA_BASE_URL}`")
    st.write(f"Wake word: `{settings.WAKE_WORD}` (voice mode in CLI)")
    st.divider()
    st.subheader("Document Assistant")
    doc_list = router.docs.list_docs()
    st.write("Docs in `data/docs/`:")
    if doc_list:
        selected = st.selectbox("Select a document", ["(none)"] + doc_list)
        if st.button("Load Document"):
            if selected != "(none)":
                msg = router.docs.load_doc_by_name(selected)
                st.success(msg)
    else:
        st.info("Add PDFs/TXT into data/docs/")

    if st.button("Clear Document"):
        st.success(router.docs.clear_loaded_doc())

prompt = st.text_input("Type your request (e.g., 'open calculator', 'summarize loaded doc')")

col1, col2 = st.columns(2)
with col1:
    send = st.button("Send")
with col2:
    clear = st.button("Clear Chat")

if clear:
    st.session_state.history = []

if send and prompt.strip():
    answer = router.handle_user_text(prompt.strip())
    st.session_state.history.append(("You", prompt.strip()))
    st.session_state.history.append(("Jarvis", answer))

st.divider()
for role, text in st.session_state.history:
    if role == "You":
        st.markdown(f"**You:** {text}")
    else:
        st.markdown(f"**Jarvis:** {text}")
