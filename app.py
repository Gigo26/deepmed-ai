import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import time

# ============================
# 1. CONFIGURACIÓN DE PÁGINA
# ============================
st.set_page_config(
    page_title="DeepMed AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================
# 2. ESTILOS CSS PRO (MODERNO)
# ============================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">

<style>
    /* --- VARIABLES DE DISEÑO --- */
    :root {
        --primary-color: #0A2647;    /* Azul Oscuro Header */
        --accent-color: #2C74B3;     /* Azul Vibrante */
        --bg-color: #F0F4F8;         /* Fondo Gris Azulado muy suave */
        --card-bg: #FFFFFF;          /* Blanco Puro */
        --text-color: #333333;
        --success: #28a745;
        --danger: #dc3545;
    }

    /* --- GLOBAL --- */
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        color: var(--text-color);
    }
    
    /* Fondo con patrón sutil */
    .stApp {
        background-color: var(--bg-color);
        background-image: radial-gradient(#d1d5db 1px, transparent 1px);
        background-size: 24px 24px;
    }

    /* --- HEADER FULL WIDTH --- */
    /* Quitamos padding nativo de Streamlit para que el header toque los bordes */
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 5rem !important;
        max-width: 1200px !important; /* Limita el ancho para que no se estire demasiado */
    }

    .custom-header {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 80px;
        background: var(--primary-color);
        z-index: 9999;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 5%;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        color: white;
    }
    
    /* Espacio para compensar el header fijo */
    .header-spacer { height: 100px; }

    /* --- TARJETAS BLANCAS (CARDS) --- */
    /* Selector para las columnas de Streamlit: Las convertimos en tarjetas */
    div[data-testid="column"] {
        background-color: var(--card-bg);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05); /* Sombra suave y elegante */
        transition: transform 0.3s ease;
        border: 1px solid rgba(0,0,0,0.02);
    }
    
    div[data-testid="column"]:hover {
        transform: translateY(-5px); /* Efecto de elevación al pasar el mouse */
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
    }

    /* Títulos de las tarjetas */
    .card-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--primary-color);
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 10px;
        border-bottom: 2px solid #f0f0f0;
        padding-bottom: 10px;
    }

    /* --- UPLOADER (ZONA DE CARGA) --- */
    /* Personalizamos el widget de carga para que se vea como en tu imagen */
    [data-testid="stFileUploader"] {
        border: 2px dashed var(--accent-color);
        border-radius: 15px;
        padding: 2rem 1rem;
        background-color: #F8FBFF; /* Fondo azul muy pálido */
        transition: all 0.3s;
    }
    [data-testid="stFileUploader"]:hover {
        background-color: #EBF5FF;
        border-color: var(--primary-color);
    }
    
    /* Texto pequeño del uploader */
    [data-testid="stFileUploader"] small {
        display: none; /* Ocultar texto por defecto feo */
    }

    /* --- BOTÓN "INICIAR ANÁLISIS" --- */
    .stButton { margin-top: 1.5rem; }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); /* Degradado moderno llamativo */
        background: linear-gradient(90deg, var(--accent-color), var(--primary-color)); /* O Azul corporativo */
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.8rem 0;
        border-radius: 12px;
        border: none;
        width: 100%;
        box-shadow: 0 4px 15px rgba(44, 116, 179, 0.4);
        transition: all 0.3s;
    }
    .stButton > button:hover {
        opacity: 0.9;
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(44, 116, 179, 0.6);
    }

    /* --- RESULTADOS PLACEHOLDER --- */
    .empty-state {
        text-align: center;
        color: #999;
        padding: 3rem 1rem;
    }
    .empty-icon { font-size: 5rem; color: #e0e0e0; margin-bottom: 1rem; }

    /* --- RESULTADOS FINALES --- */
    .result-container { text-align: center; animation: fadeIn 0.8s ease-out; }
    
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-weight: 700;
        margin-bottom: 1rem;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .status-bad { background-color: #ffebee; color: var(--danger); }
    .status-good { background-color: #e8f5e9; color: var(--success); }
    
    .probability-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin-top: 10px;
    }
    .prob-val { font-size: 1.2rem; font-weight: bold; color: var(--primary-color); }
    .prob-label { font-size: 0.8rem; color: #666; }

    /* Animación simple */
    @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

</style>

<div class="custom-header">
    <div style="display: flex; align-items: center; gap: 15px;">
        <i class="fa-solid fa-lungs" style="font-size: 2rem;"></i>
        <div>
            <h1 style="margin:0; font-size:1.5rem; font-weight:700; line-height:1.2;">DeepMed AI</h1>
            <span style="font-size:0.8rem; opacity:0.8; font-weight:300;">Sistema de Detección Temprana v3.0</span>
        </div>
    </div>
    <i class="fa-solid fa-user-md" style="font-size: 1.5rem; opacity: 0.9;"></i>
</div>
<div class="header-spacer"></div> """, unsafe_allow_html=True)

# ============================
# 3. LÓGICA IA (BACKEND)
# ============================
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
            nn.Flatten(), nn.Dropout(0.2),
            nn.Linear(56 * 14 * 14, 3000), nn.ReLU(),
            nn.Linear(3000, 1500), nn.ReLU(),
            nn.Linear(1500, 3)
        )
    def forward(self, x): return self.net(x)

@st.cache_resource
def load_model():
    try:
        model = torch.load("modelo_cnn_completo.pt", map_location="cpu")
        model.eval()
        return model
    except: return None

def preprocess_image(image):
    image = image.convert("RGB").resize((224, 224))
    tensor = transforms.Compose([transforms.ToTensor()])(image).unsqueeze(0)
    return tensor

def predict_image(model, tensor):
    categorias = ["Bengin cases", "Malignant cases", "Normal cases"]
    with torch.no_grad():
        salida = model(tensor)
        probs = F.softmax(salida, dim=1).cpu().numpy()[0]
    clase_idx = np.argmax(probs)
    return categorias[clase_idx], probs[clase_idx] * 100, probs

model = load_model()

# ============================
# 4. INTERFAZ GRÁFICA (GRID)
# ============================

# Usamos columnas con un 'gap' grande para separar las tarjetas
col1, col2 = st.columns([1, 1], gap="large")

# --- TARJETA 1: SUBIDA (IZQUIERDA) ---
with col1:
    st.markdown('<div class="card-header"><i class="fa-solid fa-cloud-upload-alt"></i> Subir Tomografía</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Arrastra tu imagen médica aquí", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        # Mostrar imagen con bordes redondeados
        st.image(image, caption="Vista Previa", use_column_width=True, output_format="PNG")
    else:
        # Espacio visual para mantener la tarjeta llena si no hay imagen
        st.markdown("<br><br>", unsafe_allow_html=True)

    # Botón grande y llamativo (DENTRO DE LA TARJETA 1)
    analyze_btn = st.button("✨ INICIAR DIAGNÓSTICO", use_container_width=True)

# --- TARJETA 2: RESULTADOS (DERECHA) ---
with col2:
    st.markdown('<div class="card-header"><i class="fa-solid fa-notes-medical"></i> Resultados IA</div>', unsafe_allow_html=True)

    if not uploaded_file or not analyze_btn:
        # ESTADO VACÍO (PLACEHOLDER)
        st.markdown("""
        <div class="empty-state">
            <i class="fa-solid fa-microscope empty-icon"></i>
            <h3 style="color: #666;">Esperando imagen...</h3>
            <p style="font-size: 0.9rem;">Sube una tomografía y presiona "Iniciar Diagnóstico" para ver el análisis detallado.</p>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        # RESULTADOS REALES
        if model:
            with st.spinner("🧠 Analizando patrones celulares..."):
                time.sleep(1.2) # Pequeña pausa dramática
                tensor = preprocess_image(image)
                clase, confianza, probs = predict_image(model, tensor)
            
            # Lógica visual de resultados
            if clase == "Malignant cases":
                badge_class = "status-bad"
                icon = "fa-radiation"
                title = "RIESGO DETECTADO"
                color = "#dc3545"
            elif clase == "Bengin cases":
                badge_class = "status-good" # Usamos verde/amarillo
                icon = "fa-shield-alt"
                title = "BENIGNO"
                color = "#28a745"
            else:
                badge_class = "status-good"
                icon = "fa-check-circle"
                title = "NORMAL"
                color = "#28a745"

            st.markdown(f"""
            <div class="result-container">
                <div class="status-badge {badge_class}">
                    <i class="fa-solid {icon}"></i> {clase}
                </div>
                
                <h1 style="color: {color}; font-size: 2.5rem; margin: 0;">{title}</h1>
                <p style="color: #666;">Confianza del Modelo: <strong>{confianza:.1f}%</strong></p>
                
                <hr style="margin: 20px 0; border: 0; border-top: 1px solid #eee;">
                
                <div style="text-align: left; font-weight: 600; margin-bottom: 10px; color: #0A2647;">Desglose de Probabilidades:</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Grid de métricas pequeñas
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f'<div class="probability-card"><div class="prob-val">{probs[0]*100:.1f}%</div><div class="prob-label">Benigno</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="probability-card"><div class="prob-val" style="color:{"red" if probs[1]>0.5 else "#0A2647"}">{probs[1]*100:.1f}%</div><div class="prob-label">Maligno</div></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="probability-card"><div class="prob-val">{probs[2]*100:.1f}%</div><div class="prob-label">Normal</div></div>', unsafe_allow_html=True)

        else:
            st.error("Error crítico: Modelo no cargado.")

# Footer simple
st.markdown("""
<div style="text-align: center; padding: 3rem; color: #aaa; font-size: 0.8rem;">
    DeepMed AI System v3.0 • Powered by PyTorch & Streamlit
</div>
""", unsafe_allow_html=True)
