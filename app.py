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
    :root {
        --primary-blue: #0A2647;
        --header-bg: linear-gradient(90deg, #0f3460 0%, #16213e 100%);
        --bg-pattern: #E8F1F5;
        --text-color: #333;
        --accent-blue: #2C74B3;
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
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* UPLOADER */
    [data-testid="stFileUploader"] {
    border: 2px dashed #2C74B3;
    border-radius: 12px;
    padding: 2.5rem 1rem;
    background-color: #f0f6ff;
    text-align: center;
    transition: 0.3s ease;
}

[data-testid="stFileUploader"]:hover {
    background-color: #e8f2ff;
}

/* Quitar el rectángulo negro */
[data-testid="stFileUploader"] section {
    background: transparent !important;
    padding: 0 !important;
    border: none !important;
    box-shadow: none !important;
}

/* Ícono de la nube */
[data-testid="stFileUploader"] section::before {
    content: "\\f0ee"; /* ícono cloud-upload de FontAwesome */
    font-family: "Font Awesome 6 Free";
    font-weight: 900;
    font-size: 3.5rem;
    color: #2C74B3;
    display: block;
    margin-bottom: 1rem;
}

/* Texto: Arrastra y suelta tu imagen aquí */
[data-testid="stFileUploader"] section > div:nth-child(1) {
    font-size: 1.2rem;
    font-weight: 600;
    color: #0A2647;
    margin-bottom: 0.3rem;
}

/* Texto de formatos soportados */
[data-testid="stFileUploader"] section > div:nth-child(2) {
    font-size: 0.9rem;
    color: #4a6fa1;
    margin-bottom: 1rem;
}

/* Botón "Browse files" personalizado */
[data-testid="stFileUploader"] button {
    border: 1px solid #2C74B3;
    background-color: white;
    color: #2C74B3;
    border-radius: 8px;
    padding: 0.5rem 1.3rem;
    font-weight: 600;
    font-size: 0.95rem;
    transition: 0.3s ease;
}

[data-testid="stFileUploader"] button:hover {
    background-color: #2C74B3;
    color: white;
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
    }
    .placeholder-icon {
        font-size: 4rem;
        color: #ddd;
        margin-bottom: 1rem;
    }

    .result-box { text-align: center; animation: fadeIn 0.5s; margin-top: 1rem; }
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

    uploaded_file = st.file_uploader("Soporta JPG, PNG, DICOM", type=["jpg", "png", "jpeg"])

    analyze_btn = st.button("Iniciar Análisis")

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Imagen cargada", width=220)


# -------- DERECHA --------
with col2:

    st.markdown('<div class="card-title"><i class="fa-solid fa-file-medical-alt"></i> Resultados del Diagnóstico</div>', 
                unsafe_allow_html=True)

    if not uploaded_file or not analyze_btn:
        st.markdown("""
        <div class="placeholder-container">
            <i class="fa-solid fa-microscope placeholder-icon"></i>
            <p>Sube una imagen y presiona "Iniciar Análisis"<br>para ver los resultados.</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        if model:
            with st.spinner("Procesando imagen..."):
                time.sleep(1)
                tensor = preprocess_image(img)
                clase, confianza, probs = predict_image(model, tensor)

            if clase == "Malignant cases":
                estado = '<div class="status-positive"><i class="fa-solid fa-circle-exclamation"></i> MALIGNO</div>'
                msg = "Se detectaron anomalías preocupantes."
            elif clase == "Bengin cases":
                estado = '<div class="status-negative" style="color:#f0ad4e;"><i class="fa-solid fa-notes-medical"></i> BENIGNO</div>'
                msg = "Tumoración benigna detectada."
            else:
                estado = '<div class="status-negative"><i class="fa-solid fa-check-circle"></i> NORMAL</div>'
                msg = "No se detectaron nódulos malignos."

            st.markdown(f"""
            <div class="result-box">
                {estado}
                <p style="color:#666;">{msg}</p>
                <h3 style="color:#0A2647;">Confianza: {confianza:.2f}%</h3>
            </div>
            """, unsafe_allow_html=True)

            st.progress(int(confianza)/100)

            colA, colB, colC = st.columns(3)
            colA.metric("Benigno", f"{probs[0]*100:.1f}%")
            colB.metric("Maligno", f"{probs[1]*100:.1f}%")
            colC.metric("Normal", f"{probs[2]*100:.1f}%")

        else:
            st.error("Error: Modelo no cargado.")

# FOOTER
st.markdown("""
<div style="text-align: center; margin-top: 3rem; color: #888; font-size: 0.8rem;">
    © 2025 DeepMed AI Solutions. Solo para fines de investigación académica.
</div>
""", unsafe_allow_html=True)
