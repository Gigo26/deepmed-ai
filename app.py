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
# 3. ESTILOS CSS (FONDO + HEADER CORREGIDO)
# ==========================================================

st.markdown("""
<style>
/* 1. Fondo general de la app */
[data-testid="stAppViewContainer"] {
    background-color: #BADFFF !important;
    background-image: radial-gradient(#000 0.5px, transparent 0.5px);
    background-size: 12px 12px;
    font-family: 'Inter', sans-serif;
}

/* 2. Ocultar el header nativo de Streamlit para que no estorbe */
header[data-testid="stHeader"] {
    background-color: transparent;
    z-index: 1; 
}

/* 3. Ajuste para que el contenido baje y no quede oculto por nuestro header fijo */
[data-testid="stAppViewContainer"] > .main {
    padding-top: 90px; 
}

/* 4. Estilo del Header Personalizado */
.custom-header {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 80px; /* Altura fija para evitar colapsos */
    
    background: linear-gradient(90deg, #00007A 0%, #6B6BDF 100%);
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    
    display: flex;
    align-items: center; /* Centrar verticalmente */
    padding-left: 3rem; /* Espacio a la izquierda */
    
    z-index: 999999; /* Z-index muy alto para asegurar que esté encima */
}

/* Contenedor del logo y textos */
.header-content {
    display: flex;
    align-items: center;
    gap: 20px;
}

/* Estilo del Ícono de Pulmones */
.icon-lungs {
    font-size: 36px; 
    color: white;
}

/* Contenedor de Textos */
.text-container {
    display: flex;
    flex-direction: column;
    justify-content: center;
}

/* Título Principal (Reemplaza al H1 para evitar conflictos de Streamlit) */
.main-title {
    font-size: 28px;
    font-weight: 800;
    color: #FFFFFF !important; /* Blanco forzado */
    text-transform: uppercase;
    letter-spacing: 1.5px;
    line-height: 1.1;
    margin: 0;
    padding: 0;
}

/* Subtítulo */
.subtitle {
    font-size: 14px;
    font-weight: 300;
    color: #E0E0E0 !important;
    margin: 0;
    padding: 0;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# 4. HEADER HTML (SIN EL MÉDICO, SOLO TÍTULO Y PULMONES)
# ==========================================================

st.markdown("""
<div class="custom-header">
    <div class="header-content">
        <!-- Ícono de pulmones -->
        <i class="fa-solid fa-lungs icon-lungs"></i>
        
        <!-- Textos -->
        <div class="text-container">
            <div class="main-title">DEEPMED AI</div>
            <div class="subtitle">Lung Cancer Detection System</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)










    st.write("El modelo está listo para recibir datos.")
