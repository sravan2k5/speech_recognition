#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import os

st.set_page_config(page_title="🎤 Speech Recognition ", layout="centered")

st.markdown("""
    <style>
        .stApp { background-color: #ffe6f0; }

        h1 { text-align: center; color: #FF4B4B; font-family: 'Segoe UI', sans-serif; }
        .main-container {
            background-color: #fff;
            padding: 30px;
            border-radius: 16px;
            box-shadow: 0 0 25px rgba(0,0,0,0.1);
            display: flex;
            justify-content: space-around;
            align-items: center;
            gap: 2rem;
            flex-wrap: wrap;
        }
        .left-box { max-width: 300px; }
        .right-box { flex: 1; min-width: 250px; }
        .footer {
            margin-top: 40px;
            text-align: center;
            font-size: 14px;
            color: #888;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>🎤 Speech Recognition System</h1>", unsafe_allow_html=True)
st.markdown("<div class='left-box'>", unsafe_allow_html=True)
st.image("E:/2d/e.jpg", caption="  ", use_column_width=True)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<div class='right-box'>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(" Upload your audio (.wav)", type=["wav"])

@st.cache_resource
def load_model():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    return processor, model

def transcribe(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    processor, model = load_model()
    inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

if uploaded_file is not None:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    st.audio("temp.wav", format="audio/wav")

    with st.spinner("Transcribing..."):
        try:
            text = transcribe("temp.wav")
            st.success("Transcription complete!")
            st.markdown("Transcribed Text")
            st.info(text)
        except Exception as e:
            st.error(f"Error: {e}")

    os.remove("temp.wav")

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)


# In[ ]:




