import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pickle
import time
import tempfile

# -------------------------------
# App Config
# -------------------------------
st.set_page_config(page_title="Instrument Detection", layout="centered")
st.title("🎵 Music Instrument Detection & Frequency Visualizer")

# -------------------------------
# Load PKL Model
import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PKL_PATH = os.path.join(BASE_DIR, "instrument_frequency_model.pkl")

with open(PKL_PATH, "rb") as f:
    model = pickle.load(f)


instrument_ranges = model["instruments"]

# -------------------------------
# Upload Audio
# -------------------------------
audio_file = st.file_uploader(
    "Upload a song file (.wav or .mp3)",
    type=["wav", "mp3"]
)
# FFT Parameters
frame_size = 2048
hop_length = 512

fig, ax = plt.subplots()
plot_area = st.pyplot(fig)

for i in range(0, len(y) - frame_size, hop_length):

    frame = y[i:i + frame_size]

    fft = np.abs(np.fft.rfft(frame))
    freqs = np.fft.rfftfreq(frame_size, 1 / sr)

    ax.clear()
    ax.plot(freqs, fft)
    ax.set_xlim(0, 5000)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Live Frequency Wave")

    plot_area.pyplot(fig)

if audio_file is not None:

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(audio_file.read())
        audio_path = tmp.name

    # Play audio
    st.audio(audio_file, format="audio/mp3")

    # Load audio
    y, sr = librosa.load(audio_path, sr=None)

    st.success("Audio loaded successfully!")

    # -------------------------------
    # FFT Parameters
    # -------------------------------
    frame_size_
