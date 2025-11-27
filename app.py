import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import time

# ============================
#     CONFIGURACIÓN DE PÁGINA
# ============================
st.set_page_config(
    page_title="DeepMed AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================
#     ESTILOS CSS (CORREGIDOS)
# ============================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">

<style>
    /* --- VARIABLES --- */
    :root {
        --primary-blue: #0A2647;
        --header-bg: #0A2647; /* Azul oscuro sólido como tu imagen */
        --bg-pattern: #E8F1F5;
        --accent-blue: #2C74B3;
    }

    /* --- GENERAL --- */
    * { font-family: 'Inter', sans-serif; }
    
    /* Fondo con patrón de puntos */
    .stApp {
        background-color: var(--bg-pattern);
        background-image: radial-gradient(#cbd5e1 1px, transparent 1px);
        background-size: 20px 20px;
    }

    /* --- 1. HEADER FULL WIDTH (CORRECCIÓN) --- */
    /* Eliminamos el padding superior estándar para pegar el header al techo */
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 2rem !important;
        max-width: 100% !important; /* Permitir ancho completo */
        padding-left: 0 !important;
        padding-right: 0 !important;
    }
    
    /* Contenedor del contenido principal (para centrarlo después del header) */
    .main-content {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }

    .custom-header {
        background-color: var(--header-bg);
        width: 100%;
        padding: 1.5rem 3rem; /* Padding interno */
        display: flex;
        align-items: center;
        justify-content: space-between;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }

    .logo-section { display: flex; align-items: center; gap: 15px; }
    .logo-icon { font-size: 2.2rem; }
    .logo-text h1 { margin: 0; font-size: 1.6rem; font-weight: 700; color: white; }
    .logo-text p { margin: 0; font-size: 0.85rem; opacity: 0.8; font-weight: 300; }

    /* --- 2. TARJETAS BLANCAS (CORRECCIÓN: Aplicado a las columnas) --- */
    /* Esto fuerza a que todo el contenido de la columna tenga fondo blanco */
    div[data-testid="column"] {
        background-color: white;
        border-radius: 12px;
        padding: 25px; /* Espacio interno para que el título no toque el borde */
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        height: 100%;
        min-height: 550px; /* Altura mínima igualada */
    }

    /* Títulos dentro de las tarjetas */
    .card-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #0A2647;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 10px;
        border-bottom: 1px solid #eee;
        padding-bottom: 10px;
    }

    /* --- 3. UPLOADER (DENTRO DE LA TARJETA) --- */
    [data-testid="stFileUploader"] {
        border: 2px dashed #2C74B3;
        border-radius: 12px;
        padding: 3rem 1rem; /* Más alto */
        background-color: #f8fbff;
        text-align: center;
    }
    
    /* --- 4. BOTÓN GRANDE (DENTRO DE LA TARJETA) --- */
    .stButton {
        margin-top: 20px;
    }
    .stButton > button {
        background-color: #6c8caf; /* Color gris azulado */
        color: white;
        border: none;
        width: 100%;
        padding: 0.8rem;
        font-size: 1.2rem; /* Letra más grande */
        border-radius: 8px;
        font-weight: 600;
        transition: 0.3s;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton > button:hover {
        background-color: #2C74B3;
        transform: translateY(-2px);
    }
    
    /* Texto de resultados */
    .placeholder-text {
        text-align: center;
        color: #888;
        margin-top: 30%;
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .placeholder-icon { font-size: 5rem; color: #e0e0e0; margin-bottom: 1rem; }

    /* Resultados finales */
    .result-box { text-align: center; margin-top: 2rem; animation: fadeIn 0.5s; }
    .status-positive { color: #d93025; font-size: 2.5rem; font-weight: 800; }
    .status-negative { color: #28a745; font-size: 2.5rem; font-weight: 800; }

</style>
""", unsafe_allow_html=True)

# ============================
#    LÓGICA DEL MODELO
# ============================
# (Mantenemos la lógica igual, solo cambiamos la vista)
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
#      HEADER (FULL WIDTH)
# ============================
st.markdown("""
<div class="custom-header">
    <div class="logo-section">
        <i class="fa-solid fa-lungs logo-icon"></i>
        <div class="logo-text">
            <h1>DeepMed AI</h1>
            <p>Lung Cancer Detection System v3.0</p>
        </div>
    </div>
    <div>
        <i class="fa-solid fa-user-doctor" style="font-size: 1.8rem;"></i>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================
#      CONTENIDO PRINCIPAL
# ============================

# Usamos un contenedor para limitar el ancho del contenido (simulando margins)
with st.container():
    # Inyectamos clase para centrar este bloque
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1], gap="large")

    # --- TARJETA IZQUIERDA: SUBIR + BOTÓN ---
    with col1:
        # TÍTULO (Ahora está DENTRO de la columna blanca por defecto gracias al CSS)
        st.markdown('<div class="card-title"><i class="fa-solid fa-upload"></i> Subir Tomografía (CT)</div>', unsafe_allow_html=True)
        
        # UPLOADER (Streamlit Widget)
        uploaded_file = st.file_uploader("Arrastra y suelta tu imagen aquí", type=["jpg", "png", "jpeg"])
        
        # VISTA PREVIA PEQUEÑA
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagen cargada", width=150)
            
        # ESPACIO
        st.markdown("<br>", unsafe_allow_html=True)

        # BOTÓN DE ANÁLISIS (Grande y dentro de la tarjeta izquierda)
        # Usamos use_container_width=True para que llene el ancho
        analyze_btn = st.button("Iniciar Análisis", use_container_width=True)

    # --- TARJETA DERECHA: RESULTADOS ---
    with col2:
        st.markdown('<div class="card-title"><i class="fa-solid fa-file-medical-alt"></i> Resultados del Diagnóstico</div>', unsafe_allow_html=True)

        if not uploaded_file or not analyze_btn:
            # PLACEHOLDER (DENTRO de la tarjeta)
            st.markdown("""
            <div class="placeholder-text">
                <i class="fa-solid fa-microscope placeholder-icon"></i>
                <p>Sube una imagen y presiona <strong>"Iniciar Análisis"</strong><br>para ver los resultados de la IA.</p>
            </div>
            """, unsafe_allow_html=True)
        
        else:
            # RESULTADOS (DENTRO de la tarjeta)
            if model is not None:
                with st.spinner('Analizando tejido...'):
                    time.sleep(1)
                    tensor = preprocess_image(image)
                    clase, confianza, probs = predict_image(model, tensor)
                
                # Definir diseño según resultado
                if clase == "Malignant cases":
                    status_html = f'<div class="status-positive"><i class="fa-solid fa-circle-exclamation"></i> POSITIVO</div>'
                    msg = "Se han detectado patrones asociados a malignidad."
                elif clase == "Bengin cases":
                    status_html = f'<div class="status-negative" style="color:#f0ad4e;"><i class="fa-solid fa-shield-virus"></i> BENIGNO</div>'
                    msg = "Nódulo detectado con características benignas."
                else:
                    status_html = f'<div class="status-negative"><i class="fa-solid fa-check-circle"></i> NEGATIVO</div>'
                    msg = "Tejido pulmonar dentro de parámetros normales."

                st.markdown(f"""
                <div class="result-box">
                    {status_html}
                    <p style="color:#666; font-size: 1.1rem; margin-top:10px;">{msg}</p>
                    <h3 style="color:#0A2647; margin-top: 20px;">Certeza: {confianza:.2f}%</h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.progress(int(confianza) / 100)
                
                # Métricas
                st.markdown("<br>", unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                c1.metric("Benigno", f"{probs[0]*100:.1f}%")
                c2.metric("Maligno", f"{probs[1]*100:.1f}%")
                c3.metric("Normal", f"{probs[2]*100:.1f}%")

            else:
                st.error("Error: Modelo no cargado.")

    st.markdown('</div>', unsafe_allow_html=True) # Cierre main-content

# FOOTER
st.markdown("""
<div style="text-align: center; margin-top: 3rem; color: #888; font-size: 0.8rem; padding-bottom: 2rem;">
    © 2025 DeepMed AI Solutions. Solo para fines de investigación académica.
</div>
""", unsafe_allow_html=True)
