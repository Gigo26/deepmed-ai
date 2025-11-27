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
# 2. CSS CON FONDO PUNTEADO Y DISEÑO COMPLETO
# ==========================================================
st.markdown("""
<style>
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body, [data-testid="stAppViewContainer"] {
    background-color: #E8F4F8 !important;
    background-image: radial-gradient(circle, #000 0.5px, transparent 0.5px) !important;
    background-size: 20px 20px !important;
    font-family: 'Inter', sans-serif;
}

/* === OCULTAR TODA LA ZONA DEL FILE UPLOADER DE STREAMLIT (DEFINITIVO) === */

/* Oculta absolutamente todo lo que Streamlit renderiza */
[data-testid="stFileUploader"] * {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
}

/* Oculta el contenedor principal */
[data-testid="stFileUploader"] {
    height: 0px !important;
    padding: 0 !important;
    margin: 0 !important;
    opacity: 0 !important;
}

/* Elimina la nube gris */
[data-testid="stFileUploaderDropzone"] {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    height: 0 !important;
}

/* Elimina textos: drag & drop, límites, etc */
[data-testid="stFileUploaderInstructions"],
[data-testid="stFileUploaderLabel"] {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
}

/* Elimina por completo el botón negro Browse Files */
[data-testid="stFileUploaderBrowseButton"] {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
}

/* Elimina contenedores internos que Streamlit añade automáticamente */
.css-1m1mtr7,
.css-1n76uvr,
.css-1uixxvy,
.css-1v0mbdj {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
}

/* ============================================================= */

/* mantengo TODO el resto igual sin tocar nada */
[data-testid="stMainBlockContainer"] {
    background-color: #E8F4F8 !important;
    background-image: radial-gradient(circle, #000 0.5px, transparent 0.5px) !important;
    background-size: 20px 20px !important;
}

/* ====== CONTENEDOR PRINCIPAL ====== */
.main-container {
    display: flex;
    gap: 40px;
    padding: 30px;
}

/* ====== COLUMNA IZQUIERDA ====== */
.left-column {
    flex: 1;
    display: flex;
    flex-direction: column;
}

/* ====== TÍTULO ====== */
.upload-title {
    font-size: 26px;
    font-weight: 900;
    color: #0A2647;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 25px;
    letter-spacing: 0.5px;
}

.upload-title i {
    font-size: 28px;
    color: #0A2647;
}

/* ====== ZONA DE UPLOAD PUNTEADA ====== */
.upload-box {
    padding: 60px 40px;
    border: 3px dashed #2C74B3;
    border-radius: 16px;
    background-color: #D4E8F0;
    text-align: center;
    transition: all 0.3s ease;
    cursor: pointer;
    margin-bottom: 20px;
    min-height: 350px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}

.upload-box:hover {
    background-color: #C5E0EB;
    border-color: #1E5A96;
    transform: translateY(-3px);
}

/* ====== ÍCONO DE NUBE ====== */
.cloud-icon {
    font-size: 80px;
    color: #2C74B3;
    margin-bottom: 20px;
    display: block;
}

/* ====== TEXTO PRINCIPAL ====== */
.upload-main-text {
    font-size: 18px;
    font-weight: 800;
    color: #000;
    margin-bottom: 8px;
    letter-spacing: 0.3px;
}

/* ====== TEXTO SECUNDARIO ====== */
.upload-subtext {
    font-size: 13px;
    color: #666;
    margin-bottom: 25px;
    font-weight: 500;
}

/* ====== BOTÓN SELECCIONAR ARCHIVO ====== */
.upload-btn-visible {
    background-color: white;
    border: 2px solid #2C74B3;
    color: #2C74B3;
    padding: 11px 32px;
    border-radius: 8px;
    font-weight: 700;
    font-size: 14px;
    display: inline-block;
    cursor: pointer;
    transition: all 0.3s ease;
}

.upload-btn-visible:hover {
    background-color: #F0F7FF;
    border-color: #1E5A96;
    transform: translateY(-2px);
}

/* ====== OCULTAR INPUT FILE ====== */
input[type="file"] {
    display: none !important;
}

/* ====== BOTÓN INICIAR ANÁLISIS (NUEVO DISEÑO) ====== */
.analyze-btn {
    width: 100%;
    padding: 18px 24px; /* MÁS ALTO */
    font-size: 18px; /* MÁS GRANDE */
    font-weight: 800;

    color: white !important;
    background: linear-gradient(90deg, #74A6D4 0%, #5C7FA8 100%); /* AZUL CLARO + OPACO */
    
    border: none;
    border-radius: 14px; /* MÁS REDONDO */
    cursor: pointer;

    transition: all 0.25s ease-in-out;
    box-shadow: 0 4px 10px rgba(100, 140, 180, 0.25);
}

/* Hover */
.analyze-btn:hover {
    background: linear-gradient(90deg, #82B6E6 0%, #6A8EB8 100%);
    transform: translateY(-2px);
    box-shadow: 0 8px 18px rgba(110, 160, 220, 0.35);
}

/* Click */
.analyze-btn:active {
    transform: scale(0.98);
}

/* ====== IMAGEN SUBIDA ====== */
.image-preview {
    margin-top: 20px;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

/* ====== COLUMNA DERECHA ====== */
.right-column {
    flex: 1;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
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
# 3. LAYOUT DE DOS COLUMNAS
# ==========================================================
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    # Título con icono
    st.markdown("""
    <div class="upload-title">
        <i class="fa-solid fa-cloud-arrow-up"></i>
        <span style="color: #0A2647; font-weight: 900;">Subir Tomografía (CT)</span>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader oculto (solo backend)
    uploaded_file = st.file_uploader(
        label="Selecciona tu archivo",
        type=["jpg", "jpeg", "png", "dcm"],
        label_visibility="collapsed",
        key="file_uploader"
    )
    
    # Zona visual de upload punteada (tu diseño)
    st.markdown("""
<div class="upload-box" onclick="document.querySelector('[data-testid=stFileUploadDropzoneInput]').click()">
    <i class="fa-solid fa-cloud-arrow-up cloud-icon"></i>
    <div class="upload-main-text">Arrastra y suelta una imagen aquí</div>
    <div class="upload-subtext">Soporta JPEG, JPG, PNG</div>
    <div class="upload-btn-visible">Seleccionar Archivo</div>
</div>
""", unsafe_allow_html=True)
    
    # Mostrar imagen si se subió
    if uploaded_file is not None:
        st.markdown('<div class="image-preview">', unsafe_allow_html=True)
        st.image(uploaded_file, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Botón Iniciar Análisis
    st.markdown('<br>', unsafe_allow_html=True)
    analyze_clicked = st.button(
        "Iniciar Análisis",
        key="analyze_btn",
        use_container_width=True
    )
    
    # Validación
    if analyze_clicked:
        if uploaded_file is None:
            st.error("⚠️ Por favor, sube una imagen primero")
        else:
            st.success("✅ Análisis iniciado...")

with col2:
    # Aquí irá la columna derecha con resultados
    st.markdown("""
    <div style="text-align: center; padding: 40px; color: #999;">
        <p style="font-size: 14px;">Los resultados aparecerán aquí</p>
    </div>
    """, unsafe_allow_html=True)
