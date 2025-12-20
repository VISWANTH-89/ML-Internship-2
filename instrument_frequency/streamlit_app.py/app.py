import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pickle
import tempfile
import os
import time

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Instrument Frequency Detector", layout="centered")
st.title("🎵 Music Instrument Detection & Frequency Visualizer")

# -------------------------------
# Load PKL Model (Safe Path)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PKL_PATH = os.path.join(BASE_DIR, "instrument_frequency_model.pkl")

with open(PKL_PATH, "rb") as f:
    model = pickle.load(f)

instrument_ranges = model["instruments"]

# -------------------------------
# Upload Audio File
# -------------------------------
audio_file = st.file_uploader(
    "Upload a song file (.wav or .mp3)",
    type=["wav", "mp3"]
)

if audio_file is not None:

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(audio_file.read())
        audio_path = tmp.name

    # Play audio
    st.audio(audio_file)

    # -------------------------------
    # Load Audio (DEFINE y HERE)
    # -------------------------------
    y, sr = librosa.load(audio_path, sr=None)
    st.success("Audio loaded successfully")

    # -------------------------------
    # FFT Parameters (DEFINE BEFORE LOOP)
    # -------------------------------
    frame_size = 2048
    hop_length = 512

    # -------------------------------
    # Streamlit Plot Area
    # -------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_area = st.pyplot(fig)

    st.subheader("🎼 Live Instrument Frequency Activity")

    # -------------------------------
    # Live Processing Loop
    # -------------------------------
    for i in range(0, len(y) - frame_size, hop_length):

        frame = y[i:i + frame_size]

        fft = np.abs(np.fft.rfft(frame))
        freqs = np.fft.rfftfreq(frame_size, 1 / sr)

        instrument_names = []
        instrument_energy = []

        # -------------------------------
        # Instrument Detection
        # -------------------------------
        for inst in instrument_ranges:
            min_hz = inst["min_hz"]
            max_hz = inst["max_hz"]

            idx = np.where((freqs >= min_hz) & (freqs <= max_hz))[0]
            energy = np.mean(fft[idx]) if len(idx) > 0 else 0

            instrument_names.append(inst["name"])
            instrument_energy.append(energy)

        # -------------------------------
        # Plot (Y-axis = Instrument Names)
        # -------------------------------
        ax.clear()
        ax.barh(instrument_names, instrument_energy)
        ax.set_xlabel("Frequency Energy (Hz-based)")
        ax.set_ylabel("Musical Instruments")
        ax.set_title("Live Detected Instrument Frequencies")

        plot_area.pyplot(fig)

        time.sleep(0.05)

    st.success("🎧 Instrument detection completed")

