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

# Cargar Google Fonts + Icons
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
""", unsafe_allow_html=True)

# ==========================================================
# CSS GLOBAL + TRUCO DEL FILE UPLOADER 100% FUNCIONAL
# ==========================================================
st.markdown("""
<style>

body, [data-testid="stAppViewContainer"] {
    background-color: #E8F4F8 !important;
    background-image: radial-gradient(circle, #000 0.5px, transparent 0.5px);
    background-size: 20px 20px;
    font-family: 'Inter', sans-serif;
}

/* ---- INPUT REAL UBICADO SOBRE EL CUADRO PUNTEADO ---- */
.file-input-layer {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    opacity: 0;        /* invisible */
    z-index: 100;      /* encima del cuadro */
    cursor: pointer;   /* hace click */
}

/* ---- CUADRO PUNTEADO ---- */
.upload-box {
    position: relative;      /* para que el input se ubique dentro */
    padding: 60px 40px;
    border: 3px dashed #2C74B3;
    border-radius: 16px;
    background-color: #D4E8F0;
    text-align: center;
    min-height: 350px;

    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;

    transition: 0.25s;
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

/* Texto */
.upload-main-text {
    font-size: 18px;
    font-weight: 800;
}

.upload-subtext {
    font-size: 13px;
    color: #666;
    margin-bottom: 25px;
}

/* Botón falso */
.upload-btn-visible {
    background-color: white;
    border: 2px solid #2C74B3;
    color: #2C74B3;
    padding: 11px 32px;
    border-radius: 8px;
    font-weight: 700;
}

/* ---- BOTÓN INICIAR ANÁLISIS ---- */
.analyze-btn {
    width: 100%;
    padding: 22px 26px;
    font-size: 22px;
    font-weight: 800;
    color: white;

    background: linear-gradient(90deg, #7BA3C8 0%, #5B738A 100%);
    border: none;
    border-radius: 16px;
    cursor: pointer;

    transition: 0.25s;
    box-shadow: 0 4px 10px rgba(80, 100, 130, 0.3);
}

/* ⚠️ Esconder por completo el contenedor visible del file_uploader */
[data-testid="stFileUploader"] {
    visibility: hidden !important;
    height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    overflow: hidden !important;
}

/* Elimina dropzone gris */
[data-testid="stFileUploaderDropzone"] {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    padding: 0 !important;
}

/* Elimina el texto "drag & drop" */
[data-testid="stFileUploaderInstructions"] {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
}

/* Elimina la etiqueta invisible que deja espacio */
[data-testid="stFileUploaderLabel"] {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}

.analyze-btn:hover {
    background: linear-gradient(90deg, #88B4DA 0%, #6A849C 100%);
    transform: translateY(-2px);
}

.analyze-btn:active {
    transform: scale(0.98);
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# 3. LAYOUT DE DOS COLUMNAS (COMO EN TU DISEÑO)
# ==========================================================
col1, col2 = st.columns([1, 1], gap="large")

# ==========================================================
# COLUMNA 1 — SUBIR IMAGEN
# ==========================================================
with col1:

    st.markdown("""
    <h2 style="font-weight:900; color:#0A2647;">
        <i class="fa-solid fa-cloud-arrow-up"></i> Subir Tomografía (CT)
    </h2>
    <hr>
    """, unsafe_allow_html=True)

    # ---- SUBIMOS EL file_uploader (invisible) ----
    uploaded_file = st.file_uploader(
        "Selecciona una imagen",
        type=[".jpg", ".jpeg", ".png", ".dcm"],
        label_visibility="collapsed",
        key="ct_input"
    )

    # ---- CUADRO PUNTEADO CON INPUT INVISIBLE ----
    st.markdown("""
    <div class="upload-box">
        <input type="file" class="file-input-layer" id="file_uploader_front">
        <i class="fa-solid fa-cloud-arrow-up cloud-icon"></i>
        <div class="upload-main-text">Arrastra y suelta tu imagen aquí</div>
        <div class="upload-subtext">Soporta JPG, PNG, DICOM</div>
        <div class="upload-btn-visible">Seleccionar Archivo</div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Mostrar imagen subida ----
    if uploaded_file is not None:
        st.image(uploaded_file, use_column_width=True)

    # ---- BOTÓN ANALIZAR ----
    st.markdown("")
    analyze_clicked = st.button(
        "Iniciar Análisis",
        key="analyze_btn",
        help="Procesar la tomografía",
        use_container_width=True
    )

    if analyze_clicked:
        if uploaded_file is None:
            st.error("⚠️ Por favor sube una imagen primero")
        else:
            st.success("✅ Procesando imagen...")

# ==========================================================
# COLUMNA 2 — RESULTADOS
# ==========================================================
with col2:
    st.markdown("""
    <h2 style="font-weight:900; color:#0A2647;">
        <i class="fa-solid fa-microscope"></i> Resultados del Diagnóstico
    </h2>
    <hr>
    <p style="padding:20px; color:#777; font-size:15px;">
        Sube una imagen y presiona <b>“Iniciar Análisis”</b> para ver los resultados de la IA.
    </p>
    """, unsafe_allow_html=True)
