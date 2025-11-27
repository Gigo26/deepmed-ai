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

# Cargar Fuentes Google e Iconos FontAwesome
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
""", unsafe_allow_html=True)

# ==========================================================
# 3. CSS DEL HEADER (CORREGIDO)
# ==========================================================
st.markdown("""
<style>
/* Ocultar header nativo de Streamlit */
[data-testid="stHeader"] {
    display: none !important;
}

/* HEADER FULL WIDTH PEGADO ARRIBA */
.custom-header {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    padding: 18px 32px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: linear-gradient(90deg, #00007A 0%, #6B6BDF 100%);
    color: white;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    z-index: 9999;
    font-family: 'Inter', sans-serif;
}

/* Ajuste correcto del contenido */
.stMainBlockContainer {
    padding-top: 110px !important;
}

/* Layout del lado izquierdo */
.header-left {
    display: flex;
    align-items: center;
    gap: 20px;
}

/* Títulos */
.header-title {
    display: flex;
    flex-direction: column;
    gap: 2px;
}

.header-title-main {
    margin: 0;
    font-size: 28px;
    font-weight: 900;
    text-transform: uppercase;
    color: white;
    letter-spacing: 1.3px;
    line-height: 1;
}

.header-subtitle {
    margin: 0;
    font-size: 13px;
    font-weight: 300;
    opacity: 0.95;
    color: #e5e5e5;
    letter-spacing: 0.5px;
}

/* Íconos */
.icon-style {
    font-size: 34px;
    color: white;
}

/* Espaciador */
.header-spacer {
    flex-grow: 1;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# 2. CSS COMPLETO Y FUNCIONAL
# ==========================================================
st.markdown("""
<style>
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

:root {
    --primary: #0A2647;
    --accent: #2C74B3;
    --purple: #7B68EE;
    --light-bg: #F4F8FF;
    --light-hover: #EBF3FF;
}

body {
    font-family: 'Inter', sans-serif;
}

/* ====== TÍTULO ====== */
.upload-title {
    font-size: 24px;
    font-weight: 900;
    color: var(--primary);
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 20px;
    letter-spacing: 0.3px;
}

.upload-title i {
    font-size: 26px;
    color: var(--primary);
}

/* ====== ZONA DE UPLOAD ====== */
.upload-container {
    margin-bottom: 20px;
}

.upload-box {
    padding: 50px 30px;
    border: 2px dashed var(--accent);
    border-radius: 16px;
    background-color: var(--light-bg);
    text-align: center;
    transition: all 0.3s ease;
    cursor: pointer;
}

.upload-box:hover {
    background-color: var(--light-hover);
    border-color: var(--purple);
}

/* Ícono de nube */
.upload-icon {
    font-size: 64px;
    color: var(--accent);
    margin-bottom: 15px;
    display: block;
}

/* Texto principal */
.upload-main-text {
    font-size: 18px;
    font-weight: 700;
    color: var(--primary);
    margin-bottom: 5px;
}

/* Texto secundario */
.upload-subtext {
    font-size: 13px;
    color: #666;
    margin-bottom: 18px;
}

/* Botón de seleccionar archivo */
.upload-btn-visible {
    background-color: white;
    border: 2px solid var(--accent);
    color: var(--accent);
    padding: 10px 24px;
    border-radius: 8px;
    font-weight: 700;
    font-size: 14px;
    display: inline-block;
    cursor: pointer;
    transition: all 0.3s ease;
}

.upload-btn-visible:hover {
    background-color: var(--light-bg);
    transform: translateY(-2px);
}

/* Ocultar input file real */
input[type="file"] {
    display: none !important;
}

/* ====== BOTÓN ANÁLISIS ====== */
.analyze-btn {
    width: 100%;
    background: linear-gradient(135deg, var(--purple) 0%, #6A5ACD 100%);
    color: white;
    font-size: 16px;
    font-weight: 700;
    padding: 14px 20px;
    border-radius: 12px;
    border: none;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(123, 104, 238, 0.3);
}

.analyze-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(123, 104, 238, 0.4);
}

.analyze-btn:active {
    transform: translateY(0);
}

/* ====== IMAGEN SUBIDA ====== */
.image-preview {
    margin-top: 20px;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# 4. HEADER HTML (CORREGIDO)
# ==========================================================
st.markdown("""
<div class="custom-header">
    <div class="header-left">
        <i class="fa-solid fa-lungs icon-style"></i>
        <div class="header-title">
            <div class="header-title-main">DEEPMED AI</div>
            <div class="header-subtitle">Lung Cancer Detection System</div>
        </div>
    </div>
    <div class="header-spacer"></div>
    <i class="fa-solid fa-user-md icon-style" title="Medical Staff"></i>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# 3. LAYOUT PRINCIPAL
# ==========================================================

# Título principal
st.markdown("""
<div class="upload-title">
    <i class="fa-solid fa-cloud-arrow-up"></i>
    Subir Tomografía (CT)
</div>
""", unsafe_allow_html=True)

# Contenedor de upload
st.markdown('<div class="upload-container">', unsafe_allow_html=True)

# File uploader (invisible)
uploaded_file = st.file_uploader(
    label="Selecciona tu archivo",
    type=["jpg", "jpeg", "png", "dcm"],
    label_visibility="collapsed"
)

# Zona visual de upload
st.markdown("""
<div class="upload-box" onclick="document.querySelector('input[type=file]').click()">
    <i class="fa-solid fa-cloud-arrow-up upload-icon"></i>
    <div class="upload-main-text">Arrastra y suelta tu imagen aquí</div>
    <div class="upload-subtext">Soporta JPG, PNG, DICOM</div>
    <div class="upload-btn-visible">Seleccionar Archivo</div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Mostrar imagen si se subió
if uploaded_file is not None:
    st.markdown('<div class="image-preview">', unsafe_allow_html=True)
    st.image(uploaded_file, use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Botón de análisis
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_clicked = st.button(
        "Iniciar Análisis",
        key="analyze_btn",
        use_container_width=True
    )

# Feedback cuando se hace clic
if analyze_clicked:
    if uploaded_file is None:
        st.error("⚠️ Por favor, sube una imagen primero")
    else:
        st.success("✅ Análisis iniciado...")
        st.info("Procesando imagen con IA...")
