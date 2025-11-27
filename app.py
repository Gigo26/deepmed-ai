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

# Fuentes e iconos
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
""", unsafe_allow_html=True)

# ==========================================================
# 3. CSS GLOBAL + ESTILO DEL FILE_UPLOADER NATIVO
# ==========================================================
st.markdown("""
<style>
body, [data-testid="stAppViewContainer"] {
    background-color: #E8F4F8 !important;
    background-image: radial-gradient(circle, #000 0.5px, transparent 0.5px);
    background-size: 20px 20px;
    font-family: 'Inter', sans-serif;
}

/* ==== QUITAR TARJETA NEGRA DEL UPLOADER ==== */
[data-testid="stFileUploader"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* ==== DARLE ESTILO DE CUADRO PUNTEADO AL DROPZONE ==== */
[data-testid="stFileUploadDropzone"] {
    border: 3px dashed #2C74B3 !important;
    border-radius: 16px !important;
    background-color: #D4E8F0 !important;
    padding: 60px 40px !important;
    min-height: 350px;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    align-items: center !important;
}

/* Ocultar icono default del uploader y usar solo texto */
[data-testid="stFileUploadDropzone"] svg {
    display: none !important;
}

/* Texto principal ("Drag and drop file here") */
[data-testid="stFileUploaderInstructions"] > div:nth-child(1) {
    font-size: 18px;
    font-weight: 800;
    color: #000;
    text-align: center;
}

/* Texto secundario (tipos soportados) */
[data-testid="stFileUploaderInstructions"] > div:nth-child(2) {
    font-size: 13px;
    color: #666;
    margin-top: 4px;
    text-align: center;
}

/* Contenedor de instrucciones centrado */
[data-testid="stFileUploaderInstructions"] {
    text-align: center !important;
}

/* Botón Browse files con estilo de "Seleccionar Archivo" */
[data-testid="stFileUploaderBrowseButton"] {
    background-color: #ffffff !important;
    border: 2px solid #2C74B3 !important;
    color: #2C74B3 !important;
    padding: 10px 30px !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    margin-top: 20px !important;
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
# 4. LAYOUT DE DOS COLUMNAS
# ==========================================================
col1, col2 = st.columns([1, 1], gap="large")

# ------------------ COLUMNA IZQUIERDA --------------------
with col1:
    st.markdown("""
    <h2 style="font-weight:900; color:#0A2647;">
        <i class="fa-solid fa-cloud-arrow-up"></i> Subir Tomografía (CT)
    </h2>
    <hr>
    """, unsafe_allow_html=True)

    # ESTE ES EL ÚNICO file_uploader QUE IMPORTA
    uploaded_file = st.file_uploader(
        "Selecciona una imagen",
        type=["jpg", "jpeg", "png", "dcm"],
        label_visibility="collapsed",
        key="ct_input"
    )

    # Mostrar imagen si se subió correctamente
    if uploaded_file is not None:
        st.image(uploaded_file, use_column_width=True)

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

# ------------------ COLUMNA DERECHA ----------------------
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
