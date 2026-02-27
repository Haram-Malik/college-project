import os
import json
import queue
import requests
import numpy as np
import streamlit as st
from vosk import Model, KaldiRecognizer
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, RTCConfiguration

# =============================
# CONFIG
# =============================
OLLAMA_MODEL = "phi3:mini"
OLLAMA_URL = "http://localhost:11434/api/generate"
VOSK_MODEL_PATH = "models/vosk"

st.set_page_config(layout="wide", page_title="Jarvis Core", page_icon="🧠")

# =============================
# HOLOGRAM STYLE
# =============================
st.markdown("""
<style>
#MainMenu, footer, header {visibility:hidden;}

.stApp {
    background: radial-gradient(circle at center, #00111a 0%, #000814 100%);
    color: #00f5ff;
    font-family: 'Segoe UI', sans-serif;
}

/* Hologram Orb */
.orb {
    width: 180px;
    height: 180px;
    border-radius: 50%;
    margin: auto;
    background: radial-gradient(circle, #00f5ff 0%, #0077b6 60%, transparent 70%);
    box-shadow: 0 0 60px #00f5ff;
    animation: pulse 3s infinite;
}

@keyframes pulse {
    0% { box-shadow: 0 0 40px #00f5ff; }
    50% { box-shadow: 0 0 80px #00f5ff; }
    100% { box-shadow: 0 0 40px #00f5ff; }
}

.glass {
    background: rgba(0, 255, 255, 0.05);
    border-radius: 15px;
    padding: 20px;
    border: 1px solid rgba(0,255,255,0.2);
    backdrop-filter: blur(10px);
}

.title {
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    margin-top: 10px;
}

.chatbox {
    max-height: 350px;
    overflow-y: auto;
}

.user { text-align:right; color:#00f5ff; }
.bot { text-align:left; color:#ff00ff; }

.wave {
    display:flex;
    gap:4px;
    height:30px;
    align-items:flex-end;
    justify-content:center;
}

.bar {
    width:5px;
    background:#00f5ff;
    animation:wave 1s infinite ease-in-out;
}

.bar:nth-child(2){animation-delay:0.1s}
.bar:nth-child(3){animation-delay:0.2s}
.bar:nth-child(4){animation-delay:0.3s}
.bar:nth-child(5){animation-delay:0.4s}

@keyframes wave {
    0%,100% {height:5px}
    50% {height:30px}
}
</style>
""", unsafe_allow_html=True)

# =============================
# SESSION
# =============================
if "history" not in st.session_state:
    st.session_state.history = []

# =============================
# FUNCTIONS
# =============================
def ask_ollama(prompt):
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120
        )
        return r.json()["response"]
    except:
        return "Ollama not running."

def execute_command(text):
    text = text.lower()
    if "open calculator" in text:
        os.system("start calc")
        return "Opening Calculator..."
    if "open google" in text:
        os.system("start https://google.com")
        return "Opening Google..."
    return ask_ollama(text)

# =============================
# HOLOGRAM CORE
# =============================
st.markdown('<div class="orb"></div>', unsafe_allow_html=True)
st.markdown('<div class="title">JARVIS CORE ONLINE</div>', unsafe_allow_html=True)

st.write("")

# =============================
# PANELS
# =============================
col1, col2, col3 = st.columns(3)

# Command Panel
with col1:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("Command Panel")
    user_input = st.text_input("Enter command", label_visibility="collapsed")
    if st.button("Execute"):
        if user_input:
            st.session_state.history.append(("You", user_input))
            reply = execute_command(user_input)
            st.session_state.history.append(("Jarvis", reply))
    st.markdown('</div>', unsafe_allow_html=True)

# Voice Panel
with col2:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("Voice Interface")

    model = Model(VOSK_MODEL_PATH)
    recognizer = KaldiRecognizer(model, 16000)
    audio_queue = queue.Queue()

    class AudioProcessor(AudioProcessorBase):
        def recv(self, frame):
            audio = frame.to_ndarray()
            mono = audio.mean(axis=1).astype(np.int16)
            audio_queue.put(mono.tobytes())
            return frame

    ctx = webrtc_streamer(
        key="speech",
        audio_processor_factory=AudioProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": []}),
        media_stream_constraints={"audio": True, "video": False},
    )

    st.markdown("""
    <div class="wave">
      <div class="bar"></div>
      <div class="bar"></div>
      <div class="bar"></div>
      <div class="bar"></div>
      <div class="bar"></div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Process Voice"):
        text = ""
        while not audio_queue.empty():
            data = audio_queue.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "")
        if text:
            st.session_state.history.append(("You (Voice)", text))
            reply = execute_command(text)
            st.session_state.history.append(("Jarvis", reply))

    st.markdown('</div>', unsafe_allow_html=True)

# Status Panel
with col3:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("System Status")
    st.write("🧠 Model:", OLLAMA_MODEL)
    st.write("🎤 Voice: Ready")
    st.write("🌐 Mode: Offline")
    st.markdown('</div>', unsafe_allow_html=True)

st.write("")

# =============================
# CHAT
# =============================
st.markdown('<div class="glass chatbox">', unsafe_allow_html=True)
for role, msg in st.session_state.history:
    if "You" in role:
        st.markdown(f'<div class="user"><b>{role}:</b> {msg}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot"><b>{role}:</b> {msg}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)