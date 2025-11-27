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
#     ESTILOS CSS (DISEÑO MEJORADO)
# ============================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">

<style>
    :root {
        --primary-blue: #0A2647;
        --header-bg: linear-gradient(90deg, #0f3460 0%, #16213e 100%);
        --bg-pattern: #E8F1F5;
        --text-color: #333;
        --accent-blue: #2C74B3;
        --light-blue-bg: #F0F8FF; /* Color de fondo estilo "bonito" */
    }

    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background-color: var(--bg-pattern);
        background-image: radial-gradient(#cbd5e1 1px, transparent 1px);
        background-size: 20px 20px;
    }

    header[data-testid="stHeader"] { visibility: hidden; }
    .block-container { padding-top: 0rem !important; padding-bottom: 2rem !important; max-width: 1200px; }

    /* HEADER */
    .custom-header {
        background: var(--header-bg);
        padding: 1rem 2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-left: -5rem;
        margin-right: -5rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }

    .logo-section { display: flex; align-items: center; gap: 10px; }
    .logo-icon { font-size: 2rem; }
    .logo-text h1 { margin: 0; font-size: 1.5rem; font-weight: 700; }
    .logo-text p { margin: 0; font-size: 0.8rem; opacity: 0.8; }

    .card-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0A2647;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* --- ESTILO DEL UPLOADER (REEMPLAZA EL NEGRO POR EL AZUL CLARO) --- */
    
    /* El contenedor principal de carga */
    [data-testid="stFileUploader"] {
        padding: 1.5rem;
        background-color: #F0F8FF; /* Fondo Azul Claro */
        border: 2px dashed #2C74B3; /* Borde Punteado Azul */
        border-radius: 15px;
        text-align: center;
    }
    
    /* Forzar que el área de drop interna también sea clara */
    [data-testid="stFileUploader"] section {
        background-color: #F0F8FF !important;
    }
    
    /* Texto "Drag and drop file here" */
    [data-testid="stFileUploader"] div span {
        color: #0A2647 !important; /* Texto Oscuro */
        font-weight: bold;
    }
    [data-testid="stFileUploader"] div small {
        color: #555 !important;
    }

    /* El botón "Browse files" dentro del uploader */
    [data-testid="stFileUploader"] button {
        border: 1px solid #2C74B3 !important;
        color: #2C74B3 !important;
        background-color: white !important;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: 0.3s;
    }
    [data-testid="stFileUploader"] button:hover {
        background-color: #E6F2FF !important;
    }

    /* BOTÓN INICIAR ANÁLISIS */
    .stButton > button {
        background: linear-gradient(90deg, #2C74B3 0%, #0A2647 100%);
        color: white;
        border: none;
        width: 100%;
        padding: 1rem;
        font-size: 1.2rem;
        border-radius: 10px;
        font-weight: 700;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        transition: 0.3s;
        margin-top: 10px;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    }

    /* PLACEHOLDER RESULTADOS */
    .placeholder-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 300px;
        color: #888;
        text-align: center;
        margin-top: 2rem;
        background: white;
        border-radius: 15px;
        box-shadow: inset 0 0 20px rgba(0,0,0,0.05);
    }
    .placeholder-icon {
        font-size: 4rem;
        color: #cbd5e1;
        margin-bottom: 1rem;
    }

    .result-box { 
        text-align: center; 
        animation: fadeIn 0.5s; 
        margin-top: 1rem; 
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
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
    return transform(image).unsqueeze(0)

def predict_image(model, tensor):
    # Nota: corregí el typo 'Bengin' a 'Benign' solo en la etiqueta visual si lo deseas, 
    # pero mantengo el string original para no romper tu lógica si depende de ese texto exacto.
    categorias = ["Bengin cases", "Malignant cases", "Normal cases"]
    with torch.no_grad():
        salida = model(tensor)
        probabilidades = F.softmax(salida, dim=1).cpu().numpy()[0]
    idx = np.argmax(probabilidades)
    return categorias[idx], probabilidades[idx] * 100, probabilidades

model = load_model()

# ============================
#      HEADER
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
#      LAYOUT SIN TARJETAS
# ============================

col1, col2 = st.columns([1, 1], gap="medium")

# -------- IZQUIERDA --------
with col1:

    st.markdown('<div class="card-title"><i class="fa-solid fa-upload"></i> Subir Tomografía (CT)</div>', 
                unsafe_allow_html=True)

    # El uploader ahora tendrá el estilo CSS definido arriba
    uploaded_file = st.file_uploader("Arrastra y suelta tu imagen aquí", type=["jpg", "png", "jpeg"])

    analyze_btn = st.button("Iniciar Análisis")

    if uploaded_file:
        img = Image.open(uploaded_file)
        # Centramos la imagen subida con un poco de estilo
        st.markdown("<div style='text-align: center; margin-top: 1rem;'>", unsafe_allow_html=True)
        st.image(img, caption="Vista previa de tomografía", width=250)
        st.markdown("</div>", unsafe_allow_html=True)


# -------- DERECHA --------
with col2:

    st.markdown('<div class="card-title"><i class="fa-solid fa-file-medical-alt"></i> Resultados del Diagnóstico</div>', 
                unsafe_allow_html=True)

    if not uploaded_file or not analyze_btn:
        st.markdown("""
        <div class="placeholder-container">
            <i class="fa-solid fa-microscope placeholder-icon"></i>
            <p><strong>Esperando imagen...</strong><br>Sube una tomografía y presiona "Iniciar Análisis".</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        if model:
            with st.spinner("Analizando tejido pulmonar..."):
                time.sleep(1) # Simulación visual
                tensor = preprocess_image(img)
                clase, confianza, probs = predict_image(model, tensor)

            if clase == "Malignant cases":
                estado = '<div class="status-positive"><i class="fa-solid fa-circle-exclamation"></i> MALIGNO</div>'
                msg = "Se detectaron anomalías compatibles con carcinoma."
            elif clase == "Bengin cases":
                estado = '<div class="status-negative" style="color:#f0ad4e;"><i class="fa-solid fa-notes-medical"></i> BENIGNO</div>'
                msg = "Tumoración benigna detectada. Seguimiento recomendado."
            else:
                estado = '<div class="status-negative"><i class="fa-solid fa-check-circle"></i> NORMAL</div>'
                msg = "Tejido pulmonar sin anomalías evidentes."

            st.markdown(f"""
            <div class="result-box">
                {estado}
                <p style="color:#666; margin-top:10px;">{msg}</p>
                <h2 style="color:#0A2647; margin-top:15px;">{confianza:.1f}% <span style="font-size:0.6em; color:#888;">de confianza</span></h2>
            </div>
            """, unsafe_allow_html=True)

            # Barra de progreso estilizada
            st.progress(int(confianza)/100)

            # Métricas detalladas
            st.markdown("<br>", unsafe_allow_html=True)
            colA, colB, colC = st.columns(3)
            colA.metric("Benigno", f"{probs[0]*100:.1f}%")
            colB.metric("Maligno", f"{probs[1]*100:.1f}%")
            colC.metric("Normal", f"{probs[2]*100:.1f}%")

        else:
            st.error("Error crítico: El archivo del modelo (modelo_cnn_completo.pt) no se encuentra.")

# FOOTER
st.markdown("""
<div style="text-align: center; margin-top: 4rem; padding-top: 1rem; border-top: 1px solid #ddd; color: #888; font-size: 0.8rem;">
    © 2025 DeepMed AI Solutions. Herramienta de apoyo al diagnóstico médico.
</div>
""", unsafe_allow_html=True)
