import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np


# ==========================================================
# 1. MODELO CNN (SE MANTIENE)
# ==========================================================

class LungCNN(nn.Module):
    def __init__(self):
        super(LungCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 112, kernel_size=3, padding=1), nn.ReLU(), nn.AvgPool2d(2),

            nn.Conv2d(112, 112, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(112, 112, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),

            nn.Conv2d(112, 112, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(112, 112, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),

            nn.Conv2d(112, 56, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(56, 56, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Dropout(0.2),

            nn.Linear(56 * 14 * 14, 3000), nn.ReLU(),
            nn.Linear(3000, 1500), nn.ReLU(),
            nn.Linear(1500, 3)
        )

    def forward(self, x):
        return self.net(x)



# ==========================================================
# 2. CONFIG PÁGINA + FUENTES
# ==========================================================

st.set_page_config(
    page_title="DeepMed AI",
    page_icon="🫁",
    layout="wide"
)

# Google Fonts + FontAwesome (importante)
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
""", unsafe_allow_html=True)

# Fondo celeste con puntos
st.markdown("""
<style>
/* Streamlit usa [data-testid="stAppViewContainer"] como body */
[data-testid="stAppViewContainer"] {
    background-color: #BADFFF !important;
    background-image: radial-gradient(#000 0.5px, transparent 0.5px);
    background-size: 12px 12px;
    font-family: 'Inter', sans-serif;
}
</style>
""", unsafe_allow_html=True)



# ==========================================================
# 3. CSS DEL HEADER
# ==========================================================

st.markdown("""
<style>

header {
    width: 100%;
    padding: 22px 40px;
    display: flex;
    justify-content: space-between;
    align-items: center;

    background: linear-gradient(90deg, #00007A 0%, #6B6BDF 100%);
    color: white;

    border-bottom: 1px solid rgba(255,255,255,0.20);
    box-shadow: 0 4px 20px rgba(0,0,0,0.20);
}

.header-left {
    display: flex;
    align-items: center;
    gap: 15px;
}

.header-title h1 {
    margin: 0;
    font-size: 30px;
    font-weight: 900;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.header-title .subtitle {
    font-size: 14px;
    opacity: 0.85;
    margin-top: -3px;
}

.header-icon {
    font-size: 36px;
    color: white;
}

</style>
""", unsafe_allow_html=True)



# ==========================================================
# 4. HEADER (SOLO ESTO, NADA MÁS)
# ==========================================================

st.markdown("""
<header>
    <div class="header-left">
        <i class="fa-solid fa-lungs" style="font-size:36px;"></i>
        <div class="header-title">
            <h1>DEEPMED AI</h1>
            <div class="subtitle">Lung Cancer Detection System</div>
        </div>
    </div>

    <i class="fa-solid fa-user-doctor header-icon"></i>
</header>
""", unsafe_allow_html=True)
