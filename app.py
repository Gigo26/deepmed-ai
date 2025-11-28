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
    font-family: 'Inter', sans-serif;
}

/* ============================================================
   ESTILO FINAL DEL UPLOADER: NUBE ARRIBA, TEXTOS CENTRADOS
   ============================================================ */

/* 1. CONTENEDOR PRINCIPAL (Caja Punteada) */
[data-testid="stFileUploaderDropzone"] {
    border: 3px dashed #2C74B3;
    background-color: #D4E8F0;
    border-radius: 20px;
    padding: 40px;
    min-height: 350px; 
    
    /* OBLIGATORIO: Flexbox Vertical */
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    align-items: center !important;
}

/* Hover effect */
[data-testid="stFileUploaderDropzone"]:hover {
    background-color: #C5E0EB;
    border-color: #1E5A96;
}

/* 2. NUBE GIGANTE CON FLECHA (Insertada en el contenedor padre) */
[data-testid="stFileUploaderDropzone"]::before {
    content: "\\f0ee";  /* Código Unicode de fa-cloud-arrow-up */
    font-family: "Font Awesome 6 Free"; /* Nombre exacto de la fuente */
    font-weight: 900; /* Peso Bold obligatorio para íconos sólidos */
    
    font-size: 90px;
    color: #2C74B3;
    display: block;
    margin-bottom: 25px; /* Espacio entre nube y texto */
    line-height: 1;
}

/* 3. TEXTO PRINCIPAL: "Arrastra y suelta..." */
/* Reemplazamos el texto 'Drag and drop' usando ::after del div interno */
/* Primero ocultamos todo lo nativo */
[data-testid="stFileUploaderDropzone"] > div {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
}

[data-testid="stFileUploaderDropzone"] svg, 
[data-testid="stFileUploaderDropzone"] small, 
[data-testid="stFileUploaderDropzone"] span {
    display: none !important;
}

/* Inyectamos nuestro Título */
[data-testid="stFileUploaderDropzone"] > div::before {
    content: "Arrastra y suelta o agrega tu imagen aquí";
    font-size: 26px;
    font-weight: 900;
    color: #0A2647;
    margin-bottom: 10px;
    text-align: center;
}

/* Inyectamos nuestro Subtítulo */
[data-testid="stFileUploaderDropzone"] > div::after {
    content: "Soporta JPG, JPEG, PNG";
    font-size: 16px;
    color: #666;
    margin-bottom: 20px;
    text-align: center;
    display: block;
}

/* 4. BOTÓN "Seleccionar Archivo" */
[data-testid="stFileUploaderDropzone"] button {
    border: 2px solid #2C74B3;
    background-color: white;
    padding: 12px 30px;
    border-radius: 10px;
    font-weight: 700;
    font-size: 16px;
    color: transparent; /* Ocultar texto original */
    position: relative;
    transition: 0.3s;
    min-width: 220px;
    margin-top: 10px; /* Separar del subtítulo */
}

/* Texto nuevo del botón */
[data-testid="stFileUploaderDropzone"] button::after {
    content: "Seleccionar Archivo";
    position: absolute;
    color: #2C74B3;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    width: 100%;
}

[data-testid="stFileUploaderDropzone"] button:hover {
    background-color: #2C74B3;
}

[data-testid="stFileUploaderDropzone"] button:hover::after {
    color: white;
}

/* 1. Estado Normal */
div.stButton > button {
    background: linear-gradient(90deg, #7BA3C8 0%, #5B738A 100%) !important;
    color: white !important;
    border: none !important;
    height: 60px !important;
    font-size: 26px !important;
    font-weight: 900 !important;
    border-radius: 70px !important;
    box-shadow: 0 4px 10px rgba(0,0,0,0.2) !important;
    transition: transform 0.2s ease !important;
}

/* 2. Estado Hover (Mouse encima) */
div.stButton > button:hover {
    background: linear-gradient(90deg, #5B738A 0%, #7BA3C8 100%) !important; /* Invertir degradado */
    color: white !important;
    transform: scale(1.02) !important; /* Pequeño efecto zoom */
    box-shadow: 0 6px 12px rgba(0,0,0,0.3) !important;
}

/* 3. Estado Active/Focus (Cuando haces clic) - QUITA EL BORDE ROJO DE STREAMLIT */
div.stButton > button:active, 
div.stButton > button:focus {
    color: white !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}

/* 4. Asegurar que el texto dentro del botón sea visible */
div.stButton > button p {
    font-size: 26px !important; 
    font-weight: 900 !important;
}

/* Estilo para el contenedor cuando YA hay imagen (Simula ser el uploader) */
.uploaded-preview {
    border: 3px solid #2C74B3; /* Borde solido en vez de punteado para indicar "listo" */
    background-color: #D4E8F0;
    border-radius: 20px;
    padding: 20px;
    min-height: 350px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    position: relative;
}

/* Botón pequeño para cerrar/cambiar imagen dentro del preview */
div.stButton > button.reset-btn {
    background-color: transparent !important;
    color: #cc0000 !important;
    border: 2px solid #cc0000 !important;
    height: auto !important;
    padding: 5px 15px !important;
    font-size: 14px !important;
    margin-top: 10px !important;
}

div.stButton > button.reset-btn:hover {
    background-color: #cc0000 !important;
    color: white !important;
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

    # 1. UPLOADER (Siempre está en el código, pero lo ocultaremos con CSS si hay archivo)
    # Usamos un key para poder limpiar el estado
    uploaded_file = st.file_uploader(
        "Sube tu tomografía", 
        type=["jpg", "jpeg", "png", "dcm"],
        key="ct_input",
        label_visibility="collapsed" # Ocultamos el label nativo pequeño
    )

    # Lógica de Visualización
    if uploaded_file is None:
        # ESTADO 1: NO HAY ARCHIVO
        # El CSS global que ya tienes se encarga de mostrar el cuadro punteado grande.
        pass
        
    else:
        # ESTADO 2: SÍ HAY ARCHIVO -> "TRUCO DE MAGIA"
        
        # A) Inyectamos CSS para OCULTAR el uploader original (el cuadro punteado vacío)
        st.markdown("""
        <style>
            [data-testid="stFileUploaderDropzone"] {
                display: none;
            }
        </style>
        """, unsafe_allow_html=True)

        # B) Mostramos nuestro contenedor "falso" con la imagen dentro
        st.markdown('<div class="uploaded-preview">', unsafe_allow_html=True)
        
        # Mostramos la imagen
        st.image(uploaded_file, use_column_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

        # C) Botón para cambiar imagen (Reset)
        # Este botón limpiará el estado para volver a mostrar el uploader
        def reset_upload():
            st.session_state["ct_input"] = None
            
        st.button("🔄 Cambiar Imagen", on_click=reset_upload, type="secondary")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # BOTÓN DE ANÁLISIS
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
        <i class="fa-solid fa-file-medical-alt"></i> Resultados del Diagnóstico
    </h2>
    <hr>
    <p style="padding:20px; color:#777; font-size:15px;">
        Sube una imagen y presiona <b>“Iniciar Análisis”</b> para ver los resultados de la IA.
    </p>
    """, unsafe_allow_html=True)
