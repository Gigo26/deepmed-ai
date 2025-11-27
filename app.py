import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import time


# ==========================================================
# 1. MODELO CNN
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


def load_model(path="modelo_cnn_completo.pt"):
    try:
        model = torch.load(path, map_location=torch.device("cpu"))
        model.eval()
        return model
    except:
        return None


def preprocess_image(file):
    image = Image.open(file).convert("RGB")
    image = image.resize((224, 224))
    transform = transforms.Compose([transforms.ToTensor()])
    return transform(image).unsqueeze(0)


def predict_image(model, tensor):
    categorias = ["Bengin cases", "Malignant cases", "Normal cases"]
    with torch.no_grad():
        salida = model(tensor)
        probabilidades = F.softmax(salida, dim=1).numpy()[0]

    idx = np.argmax(probabilidades)
    clase = categorias[idx]
    confianza = float(probabilidades[idx] * 100)
    return clase, confianza


# ==========================================================
# 2. CONFIG PÁGINA
# ==========================================================

st.set_page_config(
    page_title="DeepMed AI",
    page_icon="🫁",
    layout="wide"
)

# Fondo celeste con puntos pequeños
st.markdown("""
<style>
body {
    background-color: #BADFFF !important;
    background-image: radial-gradient(#000 0.5px, transparent 0.5px);
    background-size: 12px 12px;
}
</style>
""", unsafe_allow_html=True)


# ==========================================================
# 3. CSS GENERAL (LEFT + RIGHT)
# ==========================================================

st.markdown("""
<style>

:root {
    --primary: #0A2647;
    --accent: #2C74B3;
    --purple: #AFA0F0;
}

/* ---------------- LEFT AREA ---------------- */

.upload-title {
    font-size: 22px;
    font-weight: 850;
    color: var(--primary);
    display: flex;
    align-items: center;
    gap: 10px;
}

.upload-box {
    margin-top: 12px;
    padding: 45px 25px;
    border: 2px dashed var(--accent);
    border-radius: 15px;
    background-color: #F4F8FF;
    text-align: center;
}

.upload-box:hover {
    background-color: #EBF3FF;
}

.upload-icon {
    font-size: 58px;
    color: var(--accent);
}

.upload-main-text {
    font-size: 20px;
    font-weight: 700;
    margin-top: 10px;
}

.upload-subtext {
    font-size: 14px;
    color: #444;
    margin-top: -3px;
    margin-bottom: 12px;
}

.upload-btn-visible {
    background-color: white;
    border: 2px solid var(--accent);
    color: var(--accent);
    padding: 8px 15px;
    border-radius: 8px;
    font-weight: 700;
    display: inline-block;
    cursor: pointer;
}

/* escondemos file uploader real */
input[type="file"] {
    opacity: 0;
    position: absolute;
    top: -300px;
}

/* Botón análisis */
.analyze-btn {
    width: 100%;
    margin-top: 18px;
    background-color: var(--purple);
    color: white;
    font-size: 18px;
    font-weight: 700;
    padding: 14px;
    border-radius: 12px;
    text-align: center;
    cursor: pointer;
}

/* ---------------- RIGHT AREA ---------------- */

.result-title {
    font-size: 22px;
    font-weight: 850;
    color: var(--primary);
    display: flex;
    align-items: center;
    gap: 10px;
}

.result-placeholder {
    margin-top: 30px;
    text-align: center;
}

.result-icon {
    font-size: 75px;
    color: #CFCFCF;
}

.placeholder-text {
    font-size: 16px;
    color: #444;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# 4. LAYOUT SIN CONTENEDORES
# ==========================================================

col1, col2 = st.columns([1, 1])

# ==========================================================
# LEFT AREA
# ==========================================================
with col1:

    st.markdown("""
        <div class="upload-title">
            <i class="fa-solid fa-upload"></i>
            SUBIR TOMOGRAFÍA (CT)
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    st.markdown("""
        <div class="upload-box">
            <i class="fa-solid fa-cloud-arrow-up upload-icon"></i>

            <div class="upload-main-text">Arrastra y suelta tu imagen aquí</div>

            <div class="upload-subtext">Soporta JPG, JPEG, PNG</div>

            <div class="upload-btn-visible">Seleccionar Archivo</div>
        </div>
    """, unsafe_allow_html=True)

    if uploaded_file:
        st.image(uploaded_file, use_column_width=True)

    analyze_clicked = st.button("Iniciar Análisis")


# ==========================================================
# RIGHT AREA (placeholder)
# ==========================================================
with col2:

    st.markdown("""
        <div class="result-title">
            <i class="fa-solid fa-file-medical"></i>
            Resultados del Diagnóstico
        </div>
    """, unsafe_allow_html=True)

    placeholder_zone = st.container()

    with placeholder_zone:
        st.markdown("""
            <div class="result-placeholder">
                <div class="result-icon"><i class="fa-solid fa-microscope"></i></div>
                <div class="placeholder-text">
                    Sube una imagen y presiona "Iniciar Análisis" para ver los resultados de la IA.
                </div>
            </div>
        """, unsafe_allow_html=True)


# ==========================================================
# 5. PREDICCIÓN
# ==========================================================

model = load_model()

if analyze_clicked:

    if not uploaded_file:
        st.error("Primero sube una imagen.")
        st.stop()

    with st.spinner("Procesando imagen..."):
        tensor = preprocess_image(uploaded_file)
        clase, confianza = predict_image(model, tensor)
        time.sleep(1)

    with placeholder_zone:
        st.success("✔ Análisis completado")
        st.markdown(f"""
            <h2 style="color:#0A2647;">{clase}</h2>
            <p>Nivel de confianza:</p>
            <h3>{confianza:.2f}%</h3>
        """, unsafe_allow_html=True)
