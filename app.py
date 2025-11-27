import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np


# ==========================================================
# 1. CONFIGURACIÓN GENERAL DE LA PÁGINA
# ==========================================================

st.set_page_config(
    page_title="DeepMed AI",
    page_icon="🫁",
    layout="wide"
)

# Importar Google Fonts + FontAwesome correctamente
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
""", unsafe_allow_html=True)


# ==========================================================
# 2. CSS — CORRECCIÓN DEL HEADER (100% ANCHO, ARRIBA)
# ==========================================================

st.markdown("""
<style>

/* === RESETEO DE STREAMLIT PARA PERMITIR HEADER FULL WIDTH === */
.block-container {
    padding-top: 0rem !important;
}

/* Body de la app */
[data-testid="stAppViewContainer"] {
    background-color: #BADFFF !important;
    background-image: radial-gradient(#000 0.5px, transparent 0.5px);
    background-size: 12px 12px;
    font-family: 'Inter', sans-serif;
}

/* === HEADER REAL === */
.custom-header {
    width: 100vw;                  /* ancho total pantalla */
    margin-left: calc(-50vw + 50%); /* hack para que Streamlit no lo centre */
    background: linear-gradient(90deg, #00007A 0%, #6B6BDF 100%);
    color: white;
    padding: 22px 40px;
    display: flex;
    justify-content: space-between;
    align-items: center;

    box-shadow: 0 4px 20px rgba(0,0,0,0.20);
}

/* Contenedor izquierdo */
.header-left {
    display: flex;
    align-items: center;
    gap: 15px;
}

.header-title h1 {
    margin: 0;
    font-size: 30px;
    font-weight: 900;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.header-title .subtitle {
    font-size: 14px;
    opacity: 0.85;
    margin-top: -3px;
}

/* Ícono doctor */
.header-icon {
    font-size: 36px;
    color: white;
}

</style>
""", unsafe_allow_html=True)



# ==========================================================
# 3. HEADER HTML — YA FUNCIONA CORRECTAMENTE
# ==========================================================

st.markdown("""
<div class="custom-header">
    
    <div class="header-left">
        <i class="fa-solid fa-lungs" style="font-size:40px;"></i>

        <div class="header-title">
            <h1>DEEPMED AI</h1>
            <div class="subtitle">Lung Cancer Detection System</div>
        </div>
    </div>

    <i class="fa-solid fa-user-doctor header-icon"></i>

</div>
""", unsafe_allow_html=True)
