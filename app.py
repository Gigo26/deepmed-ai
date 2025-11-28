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
# CSS GLOBAL + TRUCO DEL FILE UPLOADER 100% FUNCIONAL
# ==========================================================
st.markdown("""
<style>
/* Fondo general */
[data-testid="stAppViewContainer"] {
    background-color: #E8F4F8;
}

/* =========================================
   ESTILOS FUERTES PARA EL FILE UPLOADER
   ========================================= */

/* 1. EL CONTENEDOR PRINCIPAL (La caja) */
[data-testid="stFileUploaderDropzone"] {
    border: 3px dashed #2C74B3;
    background-color: #D4E8F0;
    border-radius: 20px;
    padding: 40px;
    min-height: 350px; 
    
    /* TRUCO MAESTRO: Forzamos la dirección vertical en el padre */
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    align-items: center !important;
    text-align: center !important;
}

/* Hover effect */
[data-testid="stFileUploaderDropzone"]:hover {
    background-color: #C5E0EB;
    border-color: #1E5A96;
}

/* 2. FORZAR LA ESTRUCTURA INTERNA A VERTICAL */
/* Streamlit crea un div interno que por defecto es horizontal (row). 
   Aquí lo obligamos a ser vertical (column) para que la nube no se vaya de lado. */
[data-testid="stFileUploaderDropzone"] > div {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    gap: 10px; /* Espacio entre los elementos */
}

/* 3. BORRAR EL ÍCONO NATIVO PEQUEÑO Y FEO */
[data-testid="stFileUploaderDropzone"] svg {
    display: none !important;
}

/* 4. INSERTAR NUESTRA NUBE GIGANTE (FontAwesome) */
/* La insertamos en el contenedor interno para que obedezca el orden vertical */
[data-testid="stFileUploaderDropzone"] > div::before {
    content: "\f0ee"; /* fa-cloud-arrow-up */
    font-family: "Font Awesome 6 Free";
    font-weight: 900;
    font-size: 90px;    /* TAMAÑO GIGANTE */
    color: #2C74B3;     /* COLOR AZUL */
    display: block;
    margin-bottom: 20px;
    line-height: 1;
}

/* 5. TEXTO PRINCIPAL: "Arrastra y suelta..." */
/* Ocultamos el texto original y ponemos el nuestro usando ::before en el siguiente nivel */
[data-testid="stFileUploaderDropzone"] div div::before {
    content: "Arrastra y suelta tu imagen aquí";
    font-family: 'Inter', sans-serif;
    font-size: 26px;       /* TEXTO GRANDE */
    font-weight: 900;      /* NEGRITA */
    color: #0A2647;
    display: block;
    margin-bottom: 10px;
}

/* 6. SUBTÍTULO: "Soporta JPG..." */
[data-testid="stFileUploaderDropzone"] div div::after {
    content: "Soporta JPG, PNG, DICOM";
    font-family: 'Inter', sans-serif;
    font-size: 16px;       /* TAMAÑO MEDIANO */
    color: #555;           /* GRIS */
    display: block;
    margin-bottom: 20px;
}

/* Ocultar textos originales para que no se dupliquen */
[data-testid="stFileUploaderDropzone"] div div span,
[data-testid="stFileUploaderDropzone"] div div small {
    display: none !important;
}

/* 7. BOTÓN PERSONALIZADO */
[data-testid="stFileUploaderDropzone"] button {
    border: 3px solid #2C74B3;
    background-color: white;
    color: transparent; /* Texto transparente para reemplazarlo */
    padding: 15px 40px;
    border-radius: 12px;
    font-size: 1px; /* Truco para esconder el texto original */
    cursor: pointer;
    transition: 0.3s;
    width: auto;
    min-width: 250px;
    position: relative;
    margin-top: 10px;
}

/* Texto nuevo del botón */
[data-testid="stFileUploaderDropzone"] button::after {
    content: "Seleccionar Archivo";
    font-size: 18px; /* Tamaño real del texto del botón */
    font-weight: 700;
    color: #2C74B3;
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 100%;
}

/* Hover del botón */
[data-testid="stFileUploaderDropzone"] button:hover {
    background-color: #2C74B3;
}
[data-testid="stFileUploaderDropzone"] button:hover::after {
    color: white;
}

/* ESTILOS DEL BOTÓN ANALIZAR (EXTERNO) */
div.stButton > button[kind="secondary"] {
    background: linear-gradient(90deg, #7BA3C8 0%, #5B738A 100%);
    color: white;
    border: none;
    height: 60px;
    font-size: 20px;
    font-weight: bold;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)
# ==========================================================
# 4. HEADER HTML
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
# 3. LAYOUT DE DOS COLUMNAS (COMO EN TU DISEÑO)
# ==========================================================
col1, col2 = st.columns([1, 1], gap="large")

# ==========================================================
# COLUMNA 1 — SUBIR IMAGEN
# ==========================================================
with col1:
    st.markdown("""
    <h2 style="font-weight:900; color:#0A2647; margin-bottom: 5px;">
        <i class="fa-solid fa-upload"></i> Subir Tomografía (CT)
    </h2>
    <hr style="margin-top: 0px; margin-bottom: 15px;">
    """, unsafe_allow_html=True)

    # ESTE ES EL UPLOADER REAL (Ya no lo ocultaremos, lo transformaremos)
    uploaded_file = st.file_uploader(
        "Sube tu tomografía", # Texto para accesibilidad
        type=["jpg", "jpeg", "png", "dcm"],
        key="ct_input"
    )

    if uploaded_file is not None:
        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
        st.image(uploaded_file, use_column_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    analyze_clicked = st.button(
        "Iniciar Análisis",
        key="analyze_btn",
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
