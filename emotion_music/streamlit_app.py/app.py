import streamlit as st
import pickle
from gtts import gTTS
from pydub import AudioSegment
import tempfile
import os
import shutil

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Text-to-Song with Emotion Music", layout="centered")
st.title("🎤 Text-to-Song Converter with Emotion-Based Music")

# -------------------------------
# Ensure pydub finds ffmpeg
# -------------------------------
ffmpeg_path = shutil.which("ffmpeg")
if ffmpeg_path is None:
    st.warning("⚠ ffmpeg not found. Audio overlay may fail.")
else:
    AudioSegment.converter = ffmpeg_path

# -------------------------------
# Load PKL model for emotion → background music
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PKL_PATH = os.path.join(BASE_DIR, "emotion_music_model.pkl")

if not os.path.exists(PKL_PATH):
    st.error("❌ PKL file not found. Make sure emotion_music_model.pkl is in the app folder.")
else:
    with open(PKL_PATH, "rb") as f:
        model = pickle.load(f)
{
    "emotions": {
        "happy": "happy.mp3",
        "sad": "sad.mp3",
        "calm": "calm.mp3",
        "energetic": "energetic.mp3"
    }
}

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

    # Convert text to speech
    tts = gTTS(text=text, lang='en', slow=False)
    tts_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tts_temp_file.name)

    # Load background music
    bg_music_file = model["emotions"][selected_emotion]
    if not os.path.exists(bg_music_file):
        st.error(f"❌ Background music file '{bg_music_file}' not found.")
    else:
        try:
            # Load MP3 files
            bg_music = AudioSegment.from_file(bg_music_file, format="mp3").apply_gain(-10)
            tts_audio = AudioSegment.from_file(tts_temp_file.name, format="mp3")

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
        except Exception as e:
            st.error(f"❌ Error creating final song: {e}")
