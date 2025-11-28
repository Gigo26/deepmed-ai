import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import time

# ==========================================================
# 1. MODELO CNN
# ==========================================================
class LungCNN(nn.Module):
    def __init__(self):
        super(LungCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 112, kernel_size=3, padding=1), nn.ReLU(),
            nn.AvgPool2d(2),

            nn.Conv2d(112, 112, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(112, 112, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(112, 112, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(112, 112, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(112, 56, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(56, 56, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(56 * 14 * 14, 3000), nn.ReLU(),
            nn.Linear(3000, 1500), nn.ReLU(),
            nn.Linear(1500, 3)
        )

    def forward(self, x):
        return self.net(x)

# ==========================================================
# 2. TRANSFORMACIONES Y CLASES
# ==========================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

CLASSES = ["Normal", "Benigno", "Maligno"]

# ==========================================================
# 3. CARGAR MODELO
# ==========================================================
model = LungCNN()
model.eval()

# ==========================================================
# CONFIGURACIÓN DE PÁGINA
# ==========================================================
st.set_page_config(
    page_title="DeepMed AI",
    page_icon="🫁",
    layout="wide"
)

# Google fonts + Icons
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
""", unsafe_allow_html=True)

# ==========================================================
# CSS DEL HEADER + BODY + UPLOADER
# ==========================================================
st.markdown("""
<style>

/* Ocultar header nativo */
[data-testid="stHeader"] { display: none !important; }

/* HEADER */
.custom-header {
    position: fixed; top: 0; left: 0;
    width: 100%;
    padding: 18px 32px;
    background: linear-gradient(90deg, #00007A 0%, #6B6BDF 100%);
    color: white;
    display: flex;
    justify-content: space-between;
    align-items: center;
    z-index: 9999;
    font-family: 'Inter', sans-serif;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}

.header-left { display: flex; align-items: center; gap: 20px; }
.header-title { display: flex; flex-direction: column; }
.header-title-main { font-size: 28px; font-weight: 900; text-transform: uppercase; }
.header-subtitle { font-size: 13px; opacity: 0.9; }
.icon-style { font-size: 34px; }

/* Ajuste de padding */
.stMainBlockContainer { padding-top: 110px !important; }

/* Fondo general */
[data-testid="stAppViewContainer"] {
    background-color: #E8F4F8;
    font-family: 'Inter', sans-serif;
}

/* ------------------------------- */
/*        ESTILO DEL UPLOADER      */
/* ------------------------------- */
[data-testid="stFileUploaderDropzone"] {
    border: 3px dashed #2C74B3;
    background-color: #D4E8F0;
    border-radius: 20px;
    padding: 40px;
    min-height: 260px;
    display: flex !important;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}

[data-testid="stFileUploaderDropzone"]::before {
    content: "\\f0ee";
    font-family: "Font Awesome 6 Free";
    font-weight: 900;
    font-size: 65px;
    color: #2C74B3;
    margin-bottom: 18px;
}

/* Ocultar textos nativos */
[data-testid="stFileUploaderDropzone"] svg,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span {
    display: none !important;
}

[data-testid="stFileUploaderDropzone"] > div::before {
    content: "Arrastra y suelta o agrega tu imagen aquí";
    font-size: 22px; font-weight: 900; color: #0A2647;
    margin-bottom: 6px;
}

[data-testid="stFileUploaderDropzone"] > div::after {
    content: "Soporta JPG, JPEG, PNG";
    font-size: 14px; color: #666;
    margin-bottom: 16px;
}

/* Botón del uploader */
[data-testid="stFileUploaderDropzone"] button {
    border: 2px solid #2C74B3;
    background-color: white;
    padding: 12px 30px;
    border-radius: 10px;
    color: transparent;
    min-width: 200px;
    position: relative;
}

[data-testid="stFileUploaderDropzone"] button::after {
    content: "Seleccionar Archivo";
    position: absolute;
    color: #2C74B3;
    left: 50%; top: 50%;
    transform: translate(-50%, -50%);
}

[data-testid="stFileUploaderDropzone"] button:hover {
    background: #2C74B3;
}

[data-testid="stFileUploaderDropzone"] button:hover::after {
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# HEADER
# ==========================================================
st.markdown("""
<div class="custom-header">
    <div class="header-left">
        <i class="fa-solid fa-lungs icon-style"></i>
        <div class="header-title">
            <div class="header-title-main">DEEPMED AI</div>
            <div class="header-subtitle">Lung Cancer Detection System</div>
        </div>
    </div>
    <i class="fa-solid fa-user-md icon-style"></i>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# LAYOUT EN DOS COLUMNAS
# ==========================================================
col1, col2 = st.columns([1, 1], gap="large")

# ==========================================================
# COLUMNA 1 — SUBIR IMAGEN
# ==========================================================
with col1:

    st.markdown("<h2 style='font-weight:900; color:#0A2647;'>Subir Imagen</h2><hr>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        st.image(uploaded_file, use_container_width=True)

    analyze_clicked = st.button("Iniciar Análisis", use_container_width=True)

    if analyze_clicked:

        if uploaded_file is None:
            st.error("⚠️ Por favor sube una imagen primero")
        else:
            st.success("🔍 Procesando imagen...")

            start = time.time()
            image = Image.open(uploaded_file).convert("RGB")
            img_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                output = model(img_tensor)
                probs = torch.softmax(output, 1)
                conf, pred = torch.max(probs, 1)

            st.session_state["diagnosis"] = CLASSES[pred.item()]
            st.session_state["confidence"] = float(conf.item() * 100)
            st.session_state["inference"] = round(time.time() - start, 2)

# ==========================================================
# COLUMNA 2 — RESULTADOS
# ==========================================================
with col2:

    st.markdown("""
        <h2 style='font-weight:900; color:#0A2647;'>
            <i class="fa-solid fa-file-medical-alt"></i> Resultados del Diagnóstico
        </h2><hr>
    """, unsafe_allow_html=True)

    if "diagnosis" not in st.session_state:
        st.markdown("""
            <div style="text-align:center; padding:20px;">
                <i class="fa-solid fa-microscope" style="font-size:60px; color:#0A2647;"></i>
                <p style="color:#777;">Sube una imagen y presiona <b>Iniciar Análisis</b>.</p>
            </div>
        """, unsafe_allow_html=True)

    else:
        diag = st.session_state["diagnosis"]
        conf = st.session_state["confidence"]
        inf = st.session_state["inference"]

        st.markdown(f"""
        <div style="
            background:white;
            padding:25px;
            border-radius:16px;
            box-shadow:0 4px 12px rgba(0,0,0,0.1);
        ">
            <h3 style="color:#0A2647; font-weight:900; text-align:center;">
                Resultado del Modelo
            </h3>

            <p style="text-align:center; font-size:22px; font-weight:700; color:#2C74B3;">
                {diag}
            </p>

            <hr>

            <p><b>Nivel de Confianza:</b><br>
                <span style="font-size:26px; font-weight:900;">{conf:.1f}%</span>
            </p>

            <p><b>Tiempo de Inferencia:</b><br> {inf} segundos</p>
        </div>
        """, unsafe_allow_html=True)
