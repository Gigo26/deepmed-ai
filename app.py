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
#     ESTILOS CSS (REPLICA EXACTA)
# ============================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">

<style>
    /* --- VARIABLES --- */
    :root {
        --primary-blue: #0A2647;
        --header-bg: linear-gradient(90deg, #0f3460 0%, #16213e 100%);
        --card-bg: #ffffff;
        --bg-pattern: #E8F1F5;
        --text-color: #333;
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

    /* Ocultar header nativo de Streamlit */
    header[data-testid="stHeader"] { visibility: hidden; }
    .block-container { padding-top: 0rem !important; padding-bottom: 2rem !important; max-width: 1200px; }

    /* --- HEADER PERSONALIZADO --- */
    .custom-header {
        background: var(--header-bg);
        padding: 1rem 2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-left: -5rem; /* Hack para expandir a full width */
        margin-right: -5rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    
    .logo-section { display: flex; align-items: center; gap: 10px; }
    .logo-icon { font-size: 2rem; }
    .logo-text h1 { margin: 0; font-size: 1.5rem; font-weight: 700; color: white; line-height: 1.2; }
    .logo-text p { margin: 0; font-size: 0.8rem; opacity: 0.8; font-weight: 400; }

    /* --- TARJETAS (CARDS) --- */
    .css-1r6slb0, .css-12w0qpk { /* Selectores de columnas de Streamlit */
        background-color: transparent;
    }
    
    .custom-card {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        height: 600px; /* Altura fija para que se vean iguales */
        position: relative;
        display: flex;
        flex-direction: column;
    }

    .card-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0A2647;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* --- ESTILO DEL UPLOADER (EL BORDE PUNTEADO) --- */
    /* Esto disfraza el uploader nativo para que parezca el de tu imagen */
    [data-testid="stFileUploader"] {
        border: 2px dashed #2C74B3;
        border-radius: 12px;
        padding: 2rem;
        background-color: #f8fbff;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Texto "Drag and drop" nativo */
    [data-testid="stFileUploader"] section > div {
        color: #333;
    }

    /* Botón "Browse files" nativo estilizado */
    [data-testid="stFileUploader"] button {
        border: 1px solid #2C74B3;
        color: #2C74B3;
        background: white;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        margin-top: 10px;
    }

    /* --- BOTÓN GRANDE (INICIAR ANÁLISIS) --- */
    .stButton > button {
        background-color: #7895B2; /* Color gris azulado de tu imagen */
        color: white;
        border: none;
        width: 100%;
        padding: 1rem;
        font-size: 1.1rem;
        border-radius: 8px;
        font-weight: 600;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background-color: #2C74B3; /* Azul más fuerte al pasar mouse */
    }

    /* --- RESULTADOS PLACEHOLDER --- */
    .placeholder-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
        color: #888;
        text-align: center;
        margin-top: 2rem;
    }
    .placeholder-icon {
        font-size: 4rem;
        color: #ddd;
        margin-bottom: 1rem;
    }

    /* --- CLASES DE RESULTADO --- */
    .result-box { text-align: center; animation: fadeIn 0.5s; margin-top: 2rem; }
    .status-positive { color: #d93025; font-size: 2rem; font-weight: 800; }
    .status-negative { color: #28a745; font-size: 2rem; font-weight: 800; }
    
</style>
""", unsafe_allow_html=True)

# ============================
#    LÓGICA DEL MODELO
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
        
    def forward(self, x):
        return self.net(x)

@st.cache_resource
def load_model():
    try:
        model = torch.load("modelo_cnn_completo.pt", map_location="cpu")
        model.eval()
        return model
    except:
        return None

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    transform = transforms.Compose([transforms.ToTensor()])
    tensor = transform(image).unsqueeze(0)
    return tensor

def predict_image(model, tensor):
    categorias = ["Bengin cases", "Malignant cases", "Normal cases"]
    with torch.no_grad():
        salida = model(tensor)
        probabilidades = F.softmax(salida, dim=1).cpu().numpy()[0]
    clase_idx = np.argmax(probabilidades)
    return categorias[clase_idx], probabilidades[clase_idx] * 100, probabilidades

model = load_model()

# ============================
#      HEADER (HTML PURO)
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
    <div class="user-icon">
        <i class="fa-solid fa-user-doctor" style="font-size: 1.5rem;"></i>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================
#      LAYOUT PRINCIPAL
# ============================

col1, col2 = st.columns([1, 1], gap="medium")

# --- COLUMNA 1: SUBIDA (IZQUIERDA) ---
with col1:
    # Inicio Tarjeta
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    
    # Título Tarjeta
    st.markdown('<div class="card-title"><i class="fa-solid fa-upload"></i> Subir Tomografía (CT)</div>', unsafe_allow_html=True)

    # Nota: No podemos poner el icono de nube DENTRO del uploader nativo fácilmente sin romper Streamlit.
    # Pero con el CSS de arriba, el borde ya es punteado y azul.
    
    # Uploader
    uploaded_file = st.file_uploader("Soporta JPG, PNG, DICOM", type=["jpg", "png", "jpeg"])
    
    # Espaciador
    st.markdown('<div style="flex-grow: 1;"></div>', unsafe_allow_html=True)
    
    # Botón de acción
    analyze_btn = st.button("Iniciar Análisis")

    if uploaded_file:
         image = Image.open(uploaded_file)
         # Mostramos una miniatura pequeña para que no rompa el diseño
         st.image(image, caption="Imagen cargada", width=200)

    st.markdown('</div>', unsafe_allow_html=True) # Fin Tarjeta


# --- COLUMNA 2: RESULTADOS (DERECHA) ---
with col2:
    # Inicio Tarjeta
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    
    # Título Tarjeta
    st.markdown('<div class="card-title"><i class="fa-solid fa-file-medical-alt"></i> Resultados del Diagnóstico</div>', unsafe_allow_html=True)

    if not uploaded_file or not analyze_btn:
        # ESTADO "PLACEHOLDER" (Como en tu imagen)
        st.markdown("""
        <div class="placeholder-container">
            <i class="fa-solid fa-microscope placeholder-icon"></i>
            <p>Sube una imagen y presiona "Iniciar Análisis" para ver los<br>resultados de la IA.</p>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        # ESTADO CON RESULTADOS
        if model is not None:
            with st.spinner('Procesando imagen...'):
                time.sleep(1) # Simular carga para efecto visual
                tensor = preprocess_image(image)
                clase, confianza, probs = predict_image(model, tensor)
            
            # Definir estilos según resultado
            if clase == "Malignant cases":
                status_html = f'<div class="status-positive"><i class="fa-solid fa-circle-exclamation"></i> MALIGNO</div>'
                msg = "Se detectaron anomalías preocupantes."
            elif clase == "Bengin cases":
                status_html = f'<div class="status-negative" style="color:#f0ad4e;"><i class="fa-solid fa-notes-medical"></i> BENIGNO</div>'
                msg = "Tumoración benigna detectada."
            else:
                status_html = f'<div class="status-negative"><i class="fa-solid fa-check-circle"></i> NORMAL</div>'
                msg = "No se detectaron nódulos malignos."

            st.markdown(f"""
            <div class="result-box">
                {status_html}
                <p style="color:#666; margin-top:10px;">{msg}</p>
                <h3 style="color:#0A2647; margin-top: 20px;">Confianza: {confianza:.2f}%</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.progress(int(confianza) / 100)
            
            st.markdown("""
            <div style="margin-top: 30px; text-align: left; background: #f8f9fa; padding: 15px; border-radius: 8px;">
                <small><strong>Detalle de probabilidades:</strong></small>
            </div>
            """, unsafe_allow_html=True)
            
            col_res1, col_res2, col_res3 = st.columns(3)
            col_res1.metric("Benigno", f"{probs[0]*100:.1f}%")
            col_res2.metric("Maligno", f"{probs[1]*100:.1f}%")
            col_res3.metric("Normal", f"{probs[2]*100:.1f}%")

        else:
            st.error("Error: Modelo no cargado.")

    st.markdown('</div>', unsafe_allow_html=True) # Fin Tarjeta

# FOOTER
st.markdown("""
<div style="text-align: center; margin-top: 3rem; color: #888; font-size: 0.8rem;">
    © 2025 DeepMed AI Solutions. Solo para fines de investigación académica.
</div>
""", unsafe_allow_html=True)
