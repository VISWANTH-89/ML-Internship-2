import streamlit as st
import pickle
from gtts import gTTS
from pydub import AudioSegment
import tempfile
import os

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Text-to-Song with Emotion Music", layout="centered")
st.title("🎤 Text-to-Song Converter with Emotion-Based Music")

# -------------------------------
# Load PKL model for emotion → music
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PKL_PATH = os.path.join(BASE_DIR, "emotion_music_model.pkl")

with open(PKL_PATH, "rb") as f:
    model = pickle.load(f)

emotions = list(model["emotions"].keys())

# -------------------------------
# User Uploads Text File
# -------------------------------
uploaded_file = st.file_uploader("Upload a text file (.txt)", type=["txt"])

# -------------------------------
# User selects emotion
# -------------------------------
selected_emotion = st.selectbox("Select the emotion for background music", emotions)

if uploaded_file and selected_emotion:

    # Read text
    text = uploaded_file.read().decode("utf-8")

    # Convert Text to Speech
    tts = gTTS(text=text, lang='en', slow=False)

    # Save TTS temporarily
    tts_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tts_temp_file.name)

    # Load background music for selected emotion
    bg_music_file = model["emotions"][selected_emotion]
    if not os.path.exists(bg_music_file):
        st.warning(f"Background music file '{bg_music_file}' not found in the project folder.")
    else:
        bg_music = AudioSegment.from_mp3(bg_music_file).apply_gain(-10)  # Reduce volume
        tts_audio = AudioSegment.from_mp3(tts_temp_file.name)

        # Overlay TTS on background music
        final_song = bg_music.overlay(tts_audio)

        # Save final song
        final_song_file = f"{uploaded_file.name.split('.')[0]}_{selected_emotion}.mp3"
        final_song.export(final_song_file, format="mp3")

        st.success("🎵 Song generated successfully!")

        # Play final song
        st.audio(final_song_file)

        # Download final song
        with open(final_song_file, "rb") as f:
            st.download_button(
                label="Download Final Song",
                data=f,
                file_name=final_song_file,
                mime="audio/mp3"
            )
