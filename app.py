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
# 🔵 2. Configuración visual de la página (Streamlit)
# ==========================================================

st.set_page_config(
    page_title="DeepMed AI - Lung Cancer Detection",
    page_icon="🫁",
    layout="wide",
)


# ==========================================================
# 🔵 3. Inyectando CSS EXACTO de tu HTML
# ==========================================================

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

body {
    background-color: var(--light-blue);
}

header {
    background: linear-gradient(90deg, var(--primary-blue) 0%, var(--accent-blue) 100%);
    color: var(--white);
    padding: 1.5rem 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.card {
    background: var(--white);
    border-radius: var(--radius);
    padding: 2rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.05);
    min-height: 500px;
}

.upload-area {
    border: 2px dashed var(--accent-blue);
    background-color: #f8fbff;
    border-radius: var(--radius);
    padding: 30px;
    text-align: center;
    cursor: pointer;
}

.upload-area:hover {
    background-color: #eef6ff;
}

.btn-upload {
    background-color: var(--white);
    border: 1px solid var(--accent-blue);
    color: var(--accent-blue);
    padding: 0.5rem 1rem;
    border-radius: 6px;
    font-weight: 600;
}

.btn-analyze {
    background: linear-gradient(90deg, var(--accent-blue) 0%, var(--primary-blue) 100%);
    color: white;
    border-radius: var(--radius);
    padding: 14px;
    font-size: 18px;
    font-weight: 600;
    width: 100%;
    text-align: center;
    margin-top: 20px;
}

.result-box {
    padding: 20px;
    text-align: center;
}

.conf-bar {
    height: 12px;
    background: linear-gradient(90deg, var(--accent-blue), var(--success-green));
    border-radius: 10px;
}

</style>
"""

st.markdown(css, unsafe_allow_html=True)



# ==========================================================
# 🔵 4. Header idéntico al HTML
# ==========================================================

st.markdown("""
<header>
    <div style="display:flex; align-items:center; gap:15px;">
        <i class="fa-solid fa-lungs" style="font-size:32px;"></i>
        <div>
            <h1 style="margin:0;">DeepMed AI</h1>
            <div style="opacity:0.9;">Lung Cancer Detection System v3.0</div>
        </div>
    </div>

    <i class="fa-solid fa-user-doctor" style="font-size:28px;"></i>
</header>
""", unsafe_allow_html=True)



# ==========================================================
# 🔵 5. Layout principal (dos tarjetas)
# ==========================================================

col1, col2 = st.columns([1, 1])

# ==================== LEFT CARD =======================
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3><i class='fa-solid fa-upload'></i> Subir Tomografía (CT)</h3><hr>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file:
        st.image(uploaded_file, caption="Previsualización", use_column_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ==================== RIGHT CARD ======================
with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3><i class='fa-solid fa-file-medical-alt'></i> Resultados del Diagnóstico</h3><hr>", unsafe_allow_html=True)

    result_box = st.empty()

    st.markdown("</div>", unsafe_allow_html=True)


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

# ==========================================================
# 🔵 8. Footer idéntico al HTML
# ==========================================================
st.markdown("""
<br><br>
<footer style="
    text-align:center; 
    padding:1rem; 
    color:var(--text-light); 
    font-size:0.9rem;
">
    © 2025 DeepMed AI Solutions. Solo para fines de investigación académica.
</footer>
""", unsafe_allow_html=True)
