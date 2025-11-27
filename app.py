import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

# ==========================================================
# 1. MODELO CNN (SE MANTIENE IGUAL)
# ==========================================================
class LungCNN(nn.Module):
    def __init__(self):
        super(LungCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 112, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(112, 112, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(112, 112, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(112, 112, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(112, 112, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(112, 56, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(56, 56, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(56 * 14 * 14, 3000),
            nn.ReLU(),
            nn.Linear(3000, 1500),
            nn.ReLU(),
            nn.Linear(1500, 3)
        )

    def forward(self, x):
        return self.net(x)

# ==========================================================
# 2. CONFIGURACIÓN DE PÁGINA
# ==========================================================
st.set_page_config(
    page_title="DeepMed AI",
    page_icon="🫁",
    layout="wide"
)

# Google Fonts + Iconos
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
""", unsafe_allow_html=True)

# ==========================================================
# CSS GLOBAL + TRUCO DEL UPLOADER FUNCIONAL
# ==========================================================
st.markdown("""
<style>

body, [data-testid="stAppViewContainer"] {
    background-color: #E8F4F8 !important;
    background-image: radial-gradient(circle, #000 0.5px, transparent 0.5px);
    background-size: 20px 20px;
    font-family: 'Inter', sans-serif;
}

/* ---- SUBIR ARCHIVO FUNCIONAL SIN JS ---- */
.file-uploader-wrapper {
    position: relative;
    width: 100%;
}

.file-uploader-real {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 350px;
    opacity: 0;
    cursor: pointer;
    z-index: 50;
}

/* ---- CUADRO PUNTEADO ---- */
.upload-box {
    padding: 60px 40px;
    border: 3px dashed #2C74B3;
    border-radius: 16px;
    background-color: #D4E8F0;
    text-align: center;
    transition: 0.3s;
    min-height: 350px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}

.upload-box:hover {
    background-color: #C5E0EB;
    border-color: #1E5A96;
}

/* Ícono */
.cloud-icon {
    font-size: 80px;
    color: #2C74B3;
    margin-bottom: 20px;
}

/* Texto del cuadro */
.upload-main-text {
    font-size: 18px;
    font-weight: 800;
}

.upload-subtext {
    font-size: 13px;
    color: #666;
    margin-bottom: 25px;
}

/* Botón fake que se ve */
.upload-btn-visible {
    background-color: white;
    border: 2px solid #2C74B3;
    color: #2C74B3;
    padding: 11px 32px;
    border-radius: 8px;
    font-weight: 700;
}

.upload-btn-visible:hover {
    background-color: #F0F7FF;
}

/* ---- BOTÓN INICIAR ANÁLISIS ---- */
.analyze-btn {
    width: 100%;
    padding: 22px 26px;
    font-size: 22px;
    font-weight: 800;
    color: white;
    background: linear-gradient(90deg, #729DC8 0%, #54708E 100%);
    border: none;
    border-radius: 16px;
    cursor: pointer;
    transition: 0.25s;
    box-shadow: 0 4px 10px rgba(70,90,120,0.3);
}

.analyze-btn:hover {
    background: linear-gradient(90deg, #7FB0DF 0%, #5F85A4 100%);
    transform: translateY(-2px);
}

.analyze-btn:active {
    transform: scale(0.98);
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# 4. LAYOUT PRINCIPAL
# ==========================================================
st.markdown("""
<h2 style="font-weight:900; color:#0A2647;"><i class="fa-solid fa-cloud-arrow-up"></i> Subir Tomografía (CT)</h2>
""", unsafe_allow_html=True)

# ---- CUADRO DE UPLOAD FUNCIONAL ----
uploaded_file = st.file_uploader(
    "Selecciona tu archivo",
    type=["jpg", "jpeg", "png", "dcm"],
    label_visibility="collapsed",
    key="file_real_input"
)

st.markdown("""
<div class="file-uploader-wrapper">
    <div class="upload-box">
        <i class="fa-solid fa-cloud-arrow-up cloud-icon"></i>
        <div class="upload-main-text">Arrastra y suelta una imagen aquí</div>
        <div class="upload-subtext">Soporta JPEG, JPG, PNG</div>
        <div class="upload-btn-visible">Seleccionar Archivo</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---- BOTÓN ANÁLISIS ----
analyze_clicked = st.button("Iniciar Análisis", key="analyze_btn", use_container_width=True)

if analyze_clicked:
    if uploaded_file is None:
        st.error("⚠️ Por favor, sube una imagen primero")
    else:
        st.success("✅ Análisis iniciado...")

