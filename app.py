import streamlit as st

import torch

import torch.nn as nn

import torch.nn.functional as F

from torchvision import transforms

from PIL import Image

import numpy as np





# ==========================================================

# 1. MODELO CNN (SE MANTIENE)

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

# 2. CONFIG PÁGINA + FUENTES

# ==========================================================



st.set_page_config(

    page_title="DeepMed AI",

    page_icon="🫁",

    layout="wide"

)



# Google Fonts + FontAwesome (importante)

st.markdown("""

<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap" rel="stylesheet">

<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">

""", unsafe_allow_html=True)



# Fondo celeste con puntos

st.markdown("""

<style>

/* Streamlit usa [data-testid="stAppViewContainer"] como body */

[data-testid="stAppViewContainer"] {

    background-color: #BADFFF !important;

    background-image: radial-gradient(#000 0.5px, transparent 0.5px);

    background-size: 12px 12px;

    font-family: 'Inter', sans-serif;

}

</style>

""", unsafe_allow_html=True)

# ==========================================================
# 3. CSS DEL HEADER (CORREGIDO PARA FULL WIDTH)
# ==========================================================

st.markdown("""
<style>
/* Ocultar el header default de Streamlit (la línea de colores y menú hamburguesa) si molesta */
/* o ajustarlo para que no tape nuestro header */
[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);
    z-index: 1; 
}

/* Estilo para nuestro Header Personalizado */
.custom-header {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw; /* 100% del ancho de la ventana */
    padding: 1rem 3rem; /* Espaciado interno */
    
    display: flex;
    justify-content: space-between;
    align-items: center;

    background: linear-gradient(90deg, #00007A 0%, #6B6BDF 100%);
    color: white;
    
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    z-index: 9999; /* Para que quede por encima de todo */
}

/* Ajuste para que el contenido de la app no quede oculto detrás del header fijo */
[data-testid="stAppViewContainer"] > .main {
    padding-top: 80px; 
}

.header-left {
    display: flex;
    align-items: center;
    gap: 20px;
}

.header-title h1 {
    margin: 0;
    font-size: 28px;
    font-weight: 800;
    color: white !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    line-height: 1.2;
}

.header-title .subtitle {
    font-size: 14px;
    font-weight: 300;
    opacity: 0.9;
    color: #E0E0E0;
}

/* Estilo específico para los íconos */
.icon-style {
    font-size: 32px; 
    color: white;
}
</style>
""", unsafe_allow_html=True)


# ==========================================================
# 4. HEADER HTML (CORREGIDO CON ÍCONO VISIBLE)
# ==========================================================

st.markdown("""
<div class="custom-header">
    <div class="header-left">
        <!-- Ícono de pulmones -->
        <i class="fa-solid fa-lungs icon-style"></i>
        
        <div class="header-title">
            <h1>DEEPMED AI</h1>
            <div class="subtitle">Lung Cancer Detection System</div>
        </div>
    </div>

    <!-- Ícono de doctor cambiado a 'fa-user-md' que es más compatible -->
    <i class="fa-solid fa-user-md icon-style" title="Medical Staff"></i>
</div>
""", unsafe_allow_html=True)
