import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import base64
import time


# ==========================================================
# 🔵 1. Cargar tu modelo CNN
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
            nn.Linear(1500, 3)  # 3 clases
        )
        
    def forward(self, x):
        return self.net(x)


def load_model(path="modelo_cnn_completo.pt"):
    try:
        model = torch.load(path, map_location=torch.device("cpu"))
        model.eval()
        return model
    except Exception as e:
        print("ERROR cargando modelo:", e)
        return None


def preprocess_image(file):
    image = Image.open(file).convert("RGB")
    image = image.resize((224, 224))

    transform = transforms.Compose([transforms.ToTensor()])
    tensor = transform(image).unsqueeze(0)
    return tensor


def predict_image(model, tensor):
    categorias = ["Bengin cases", "Malignant cases", "Normal cases"]

    with torch.no_grad():
        salida = model(tensor)
        probabilidades = F.softmax(salida, dim=1).numpy()[0]

    idx = np.argmax(probabilidades)
    clase = categorias[idx]
    confianza = float(probabilidades[idx] * 100)

    return clase, confianza, probabilidades.tolist()



# ==========================================================
# 🔵 2. CONFIG PAGINA
# ==========================================================

st.set_page_config(
    page_title="DeepMed AI - Lung Cancer Detection",
    page_icon="🫁",
    layout="wide",
)

# Fondo de la página (color #BADFFF)
page_bg = """
<style>
body {
    background-color: #BADFFF !important;
}
</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

# ============================
#        CSS GENERAL
# ============================

css = """
<style>
:root {
    --primary-blue: #0A2647;
    --accent-blue: #2C74B3;
    --light-blue: #E8F1F5;
    --success-green: #28a745;
    --white: #ffffff;
    --text-dark: #333333;
    --text-light: #666666;
    --radius: 12px;
}

/* FONDO GENERAL */
body {
    background-color: var(--light-blue);
    font-family: 'Inter', sans-serif;
}

</style>
"""
st.markdown(css, unsafe_allow_html=True)

# ============================
#        CSS HEADER
# ============================

header_css = """
<style>
header {
    width: 100%;
    padding: 20px 40px;
    display: flex;
    justify-content: space-between;
    align-items: center;

    background: linear-gradient(90deg, #00007A 0%, #6B6BDF 100%);
    color: white;

    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    border-bottom: 1px solid rgba(255,255,255,0.15);
}

.header-left {
    display: flex;
    align-items: center;
    gap: 15px;
}

.header-title h1 {
    margin: 0;
    font-size: 28px;
    font-weight: 900;
    text-transform: uppercase;
    color: white;
}

.header-title .subtitle {
    margin: 0;
    margin-top: -3px;
    font-size: 14px;
    opacity: 0.85;
    color: white;
}

.header-icon {
    font-size: 35px;
    color: white;
}
</style>
"""
st.markdown(header_css, unsafe_allow_html=True)

# ============================
#        CSS LEFT
# ============================

upload_css = """
<style>

.upload-card {
    background-color: white;
    padding: 20px 25px;
    border-radius: 15px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.08);
    width: 100%;
}

/* Título */
.upload-title {
    font-size: 20px;
    font-weight: 800;
    text-transform: uppercase;
    color: #0A2647;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Caja punteada */
.upload-box {
    margin-top: 20px;
    padding: 45px 25px;
    border: 2px dashed #2C74B3;
    border-radius: 15px;
    background-color: #F4F8FF;
    text-align: center;
}

.upload-box:hover {
    background-color: #EBF3FF;
}

/* Ícono nube */
.upload-icon {
    font-size: 58px;
    color: #2C74B3;
}

/* Texto principal */
.upload-main-text {
    font-size: 20px;
    font-weight: 700;
    margin-top: 12px;
    color: #000;
}

/* Subtexto */
.upload-subtext {
    font-size: 14px;
    color: #444;
    margin-top: -5px;
    margin-bottom: 15px;
}

/* Botón visual de seleccionar archivo */
.upload-btn-visible {
    background-color: white;
    border: 2px solid #2C74B3;
    color: #2C74B3;
    padding: 8px 15px;
    border-radius: 8px;
    font-weight: 700;
    display: inline-block;
    cursor: pointer;
}

/* Oculta el file uploader real */
input[type="file"] {
    opacity: 0;
    position: absolute;
    top: -100px;
}

</style>
"""
st.markdown(upload_css, unsafe_allow_html=True)

# ============================
#          CSS RIGHT
# ============================

right_css = """
<style>

.result-card {
    width: 100%;
    background-color: white;
    border-radius: 15px;
    padding: 25px 30px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.05);
    min-height: 520px;
}

.result-title {
    font-size: 22px;
    font-weight: 800;
    color: #0A2647;
    display: flex;
    align-items: center;
    gap: 10px;
}

.result-line {
    width: 100%;
    height: 1px;
    background-color: #E0E0E0;
    margin: 12px 0 25px 0;
}

.result-placeholder {
    margin-top: 40px;
    text-align: center;
    color: #7A7A7A;
}

.result-icon {
    font-size: 70px;
    color: #CFCFCF;   /* gris claro */
    margin-bottom: 10px;
}

.placeholder-text {
    font-size: 16px;
    color: #777;
    line-height: 1.4;
}

</style>
"""
st.markdown(right_css, unsafe_allow_html=True)

# ============================
#          CSS FOOTER
# ============================

footer_css = """
<style>
.app-footer {
    width: 100%;
    text-align: center;
    padding: 15px 0;
    margin-top: 30px;
    font-size: 13px;
    color: #666;
}
</style>
"""
st.markdown(footer_css, unsafe_allow_html=True)

# ============================
#          HEADER
# ============================

st.markdown("""
<header>
    <div class="header-left">
        <i class="fa-solid fa-lungs" style="font-size:35px; color:white;"></i>
        <div class="header-title">
            <h1>DEEPMED AI</h1>
            <div class="subtitle">Lung Cancer Detection System</div>
        </div>
    </div>

    <i class="fa-solid fa-user-doctor header-icon"></i>
</header>
""", unsafe_allow_html=True)


# =============================
#           LAYOUT  
# =============================

col1, col2 = st.columns([1, 1])

# ==================== LEFT CARD =======================
with col1:

    # CARD contenedor
    st.markdown("<div class='upload-card'>", unsafe_allow_html=True)

    # Título
    st.markdown("""
        <div class="upload-title">
            <i class="fa-solid fa-upload"></i>
            SUBIR TOMOGRAFÍA (CT)
        </div>
    """, unsafe_allow_html=True)

    # Uploader funcional de Streamlit (invisible)
    uploaded_file = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key="uploader_real"
    )

    # Caja punteada (estética)
    st.markdown("""
        <div class="upload-box">
            <i class="fa-solid fa-cloud-arrow-up upload-icon"></i>

            <div class="upload-main-text">Arrastra y suelta tu imagen aquí</div>

            <div class="upload-subtext">Soporta JPG, JPEG, PNG</div>

            <div class="upload-btn-visible">Seleccionar Archivo</div>
        </div>
    """, unsafe_allow_html=True)

    # Previsualización
    if uploaded_file:
        st.image(uploaded_file, caption="", use_column_width=True)

    # Botón Iniciar Análisis
    st.markdown("""
        <div class="analyze-btn">
            Iniciar Análisis
        </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ==================== RIGHT CARD ======================
with col2:
    st.markdown("""
    <div class="result-card">

        <!-- Título -->
        <div class="result-title">
            <i class="fa-solid fa-file-medical"></i>
            Resultados del Diagnóstico
        </div>

        <div class="result-line"></div>

        <!-- Placeholder inicial -->
        <div class="result-placeholder">
            <div class="result-icon">
                <i class="fa-solid fa-microscope"></i>
            </div>

            <div class="placeholder-text">
                Sube una imagen y presiona "Iniciar Análisis" para ver los resultados de la IA.
            </div>
        </div>

    </div>
    """, unsafe_allow_html=True)

# ==========================================================
# 🔵 6. Botón de análisis
# ==========================================================

st.markdown("<br>", unsafe_allow_html=True)

analyze = st.button("Iniciar Análisis", use_container_width=True)

model = load_model()


# ==========================================================
# 🔵 7. Ejecución de predicción
# ==========================================================

if analyze:

    if not uploaded_file:
        st.error("Primero sube una imagen.")
        st.stop()

    with st.spinner("Procesando imagen..."):
        tensor = preprocess_image(uploaded_file)
        clase, confianza, probs = predict_image(model, tensor)
        time.sleep(1)

    # Mostrar resultados
    result_box.markdown(f"""
    <div class='result-box'>
        <div style="background:#e6f4ea; color:#28a745; padding:10px; border-radius:50px; display:inline-block; font-weight:700;">
            <i class="fa-solid fa-check-circle"></i> Análisis Completado
        </div>

        <h2 style="color:var(--primary-blue); margin-top:15px;">{clase}</h2>
        <p style="color:var(--text-light);">Nivel de confianza</p>

        <strong>{confianza:.2f}%</strong>

        <div class="confidence-bar-container" style="margin-top:10px;">
            <div class="conf-bar" style="width:{confianza}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==================================
#             FOOTER
# ==================================
st.markdown("""
<div class="app-footer">
    © 2025 DeepMed AI Solutions. Solo para fines de investigación académica.
</div>
""", unsafe_allow_html=True)
