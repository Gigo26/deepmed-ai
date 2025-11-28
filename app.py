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
   ESTILOS AVANZADOS DEL FILE UPLOADER v3 (CORREGIDO)
   ========================================= */

/* 1. CONTENEDOR PRINCIPAL (La caja punteada) */
/* Configuramos Flexbox vertical para que todo se apile una cosa sobre otra */
[data-testid="stFileUploaderDropzone"] {
    border: 4px dashed #2C74B3; /* Borde un poco más grueso */
    background-color: #D4E8F0;
    border-radius: 20px;
    padding: 40px 30px;
    min-height: 350px; /* Altura para que respire */
    
    display: flex;
    flex-direction: column; /* APARECERÁ UNO DEBAJO DEL OTRO */
    justify-content: center;
    align-items: center;
    text-align: center;
    transition: 0.3s;
}

/* Hover del contenedor */
[data-testid="stFileUploaderDropzone"]:hover {
    background-color: #C5E0EB;
    border-color: #1E5A96;
}

/* 2. EL ÍCONO DE LA NUBE (FontAwesome) */
/* Se inserta ANTES (::before) de todo el contenido, por eso queda arriba */
[data-testid="stFileUploaderDropzone"]::before {
    content: "\f0ee"; /* Código del ícono fa-cloud-arrow-up */
    font-family: "Font Awesome 6 Free";
    font-weight: 900;
    font-size: 80px;    /* <--- TAMAÑO GIGANTE */
    color: #2C74B3;
    display: block;
    margin-bottom: 20px; /* Espacio debajo de la nube */
    line-height: 1;
}

/* 3. OCULTAR ELEMENTOS NATIVOS DE STREAMLIT */
/* Ocultamos el ícono pequeño por defecto y los textos originales */
[data-testid="stFileUploaderDropzone"] > div > div > svg, /* El ícono nativo pequeño */
[data-testid="stFileUploaderDropzone"] div div span,
[data-testid="stFileUploaderDropzone"] div div small {
    display: none !important;
}

/* 4. TÍTULO PRINCIPAL: "Arrastra y suelta..." */
/* Inyectamos este texto y lo hacemos grande */
[data-testid="stFileUploaderDropzone"] div div::before {
    content: "Arrastra y suelta tu imagen aquí";
    font-family: 'Inter', sans-serif;
    font-size: 28px;       /* <--- TEXTO MUCHO MÁS GRANDE */
    font-weight: 900;      /* Extra negrita */
    color: #0A2647;
    display: block;
    margin-bottom: 10px;
    line-height: 1.2;
}

/* 5. SUBTÍTULO: "Soporta JPG, PNG..." */
[data-testid="stFileUploaderDropzone"] div div::after {
    content: "Soporta JPG, PNG, DICOM";
    font-family: 'Inter', sans-serif;
    font-size: 18px;       /* <--- SUBTÍTULO MÁS GRANDE */
    font-weight: 500;
    color: #555555;        /* Un gris un poco más oscuro para contraste */
    display: block;
    margin-bottom: 25px;   /* Espacio antes del botón */
}

/* 6. ESTILO DEL BOTÓN */
[data-testid="stFileUploaderDropzone"] button {
    border: 3px solid #2C74B3;
    background-color: white;
    color: transparent; /* Hacemos transparente el texto original "Browse files" */
    padding: 14px 35px;
    border-radius: 10px;
    font-weight: 700;
    font-size: 18px;
    cursor: pointer;
    transition: 0.3s;
    position: relative;
    min-width: 250px; /* Botón más ancho */
}

/* Texto nuevo del botón superpuesto */
[data-testid="stFileUploaderDropzone"] button::after {
    content: "Seleccionar Archivo";
    position: absolute;
    color: #2C74B3;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%); /* Centrado perfecto */
    width: 100%;
}

/* Hover del botón */
[data-testid="stFileUploaderDropzone"] button:hover {
    background-color: #2C74B3;
    border-color: #2C74B3;
}
[data-testid="stFileUploaderDropzone"] button:hover::after {
    color: white;
}

/* ESTILOS DEL BOTÓN ANALIZAR (Global) */
div.stButton > button[kind="secondary"] {
    background: linear-gradient(90deg, #7BA3C8 0%, #5B738A 100%);
    color: white;
    border: none;
    height: 65px;
    font-size: 22px;
    font-weight: 800;
    border-radius: 14px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
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
