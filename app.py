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
    page_title="DeepMed AI - Lung Cancer Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================
#     ESTILOS CSS (DISEÑO)
# ============================
# Aquí traducimos tu CSS del HTML a Streamlit
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">

<style>
    /* Variables Globales */
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

    /* Fuente General */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Fondo de la App */
    .stApp {
        background-color: var(--light-blue);
        background-image: radial-gradient(#dbe9f6 1px, transparent 1px);
        background-size: 20px 20px;
    }

    /* Ocultar elementos nativos molestos */
    header {visibility: hidden;}
    .block-container {padding-top: 1rem !important; padding-bottom: 1rem !important;}

    /* HEADER PERSONALIZADO */
    .custom-header {
        background: linear-gradient(90deg, var(--primary-blue) 0%, var(--accent-blue) 100%);
        padding: 1.5rem 2rem;
        border-radius: var(--radius);
        color: white;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .header-title h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
        color: white !important;
    }
    .header-subtitle {
        font-size: 0.9rem;
        opacity: 0.9;
        font-weight: 300;
    }

    /* TARJETAS (CARDS) */
    .custom-card {
        background: var(--white);
        padding: 2rem;
        border-radius: var(--radius);
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        height: 100%;
        min-height: 500px;
    }

    .card-header {
        border-bottom: 1px solid #eee;
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
        color: var(--primary-blue);
        font-weight: 600;
        font-size: 1.2rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* Estilo del File Uploader nativo de Streamlit */
    [data-testid="stFileUploader"] {
        border: 2px dashed var(--accent-blue);
        border-radius: var(--radius);
        padding: 2rem;
        background-color: #f8fbff;
        text-align: center;
        transition: all 0.3s;
    }
    [data-testid="stFileUploader"]:hover {
        background-color: #eef6ff;
        border-color: var(--primary-blue);
    }
    
    /* Botones */
    .stButton > button {
        background: linear-gradient(90deg, var(--accent-blue) 0%, var(--primary-blue) 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: var(--radius);
        font-weight: 600;
        width: 100%;
        transition: opacity 0.3s;
    }
    .stButton > button:hover {
        opacity: 0.9;
        color: white;
        border: none;
    }

    /* RESULTADOS */
    .result-badge {
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 1rem;
        font-size: 0.9rem;
    }
    .badge-success { background-color: #e6f4ea; color: var(--success-green); }
    .badge-danger { background-color: #fce8e6; color: #d93025; }
    
    .big-prediction {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }

    .details-list {
        list-style: none;
        padding: 0;
        margin-top: 2rem;
        font-size: 0.95rem;
    }
    .details-list li {
        display: flex;
        justify-content: space-between;
        padding: 12px 0;
        border-bottom: 1px solid #f0f0f0;
        color: var(--text-dark);
    }
    .details-list span { font-weight: 600; color: var(--primary-blue); }

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
    except Exception as e:
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

# Cargar modelo
model = load_model()

# ============================
#      HEADER (HTML PURO)
# ============================
st.markdown("""
<div class="custom-header">
    <div class="header-title">
        <div style="display: flex; align-items: center; gap: 15px;">
            <i class="fa-solid fa-lungs" style="font-size: 2rem;"></i>
            <div>
                <h1>DeepMed AI</h1>
                <div class="header-subtitle">Lung Cancer Detection System v3.0</div>
            </div>
        </div>
    </div>
    <div>
        <i class="fa-solid fa-user-doctor" style="font-size: 1.5rem;"></i>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================
#      LAYOUT PRINCIPAL (nuevo limpio)
# ============================

# Ocultar uploader original de Streamlit
hide_uploader_css = """
<style>
[data-testid="stFileUploader"] {
    opacity: 0;
    height: 0px;
    position: absolute;
    z-index: -1;
}
</style>
"""
st.markdown(hide_uploader_css, unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

# -------------------------------------------------------
#  🩻   COLUMNA IZQUIERDA – SUBIR TOMOGRAFÍA (UI limpia)
# -------------------------------------------------------
with col1:

    st.markdown("""
    <h3 style="color:#0A2647; margin-bottom:15px;">
        <i class="fa-solid fa-upload"></i> Subir Tomografía (CT)
    </h3>
    <hr style="margin-top:-5px; margin-bottom:20px; border-color:#d0d7e1;">
    """, unsafe_allow_html=True)

    # CUADRO PUNTEADO
    st.markdown("""
    <div style="
        border: 2px dashed #2C74B3;
        border-radius: 14px;
        padding: 50px 20px;
        background-color:#f8fbff;
        text-align:center;
        margin-bottom:22px;
    ">
        <i class="fa-solid fa-cloud-arrow-up" style="font-size:4rem; color:#2C74B3;"></i>
        <h4 style="margin-top:15px; font-weight:700; color:#0A2647;">
            Arrastra y suelta tu imagen aquí
        </h4>
        <p style="color:#777; font-size:0.9rem; margin-top:5px;">
            Soporta JPG, PNG, DICOM
        </p>
    </div>
    """, unsafe_allow_html=True)

    # UPLOADER OCULTO
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

    # BOTÓN PERSONALIZADO
    select_btn = st.button("Seleccionar Archivo", use_container_width=True)

    # Cuando el usuario presiona el botón → activamos el uploader real
    if select_btn:
        st.session_state["trigger_upload"] = True

    if "trigger_upload" in st.session_state and st.session_state["trigger_upload"]:
        uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        st.session_state["trigger_upload"] = False

    # Vista previa
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Vista previa de la tomografía", use_column_width=True)

    # BOTÓN ANALIZAR
    analizar = st.button("Iniciar Análisis", use_container_width=True)


# -------------------------------------------------------
#  ⚕️   COLUMNA DERECHA – RESULTADOS
# -------------------------------------------------------
with col2:

    st.markdown("""
    <h3 style="color:#0A2647; margin-bottom:15px;">
        <i class="fa-solid fa-file-medical-alt"></i> Resultados del Diagnóstico
    </h3>
    <hr style="margin-top:-5px; margin-bottom:20px; border-color:#d0d7e1;">
    """, unsafe_allow_html=True)

    if model is None:
        st.error("Error: No se encontró el archivo 'modelo_cnn_completo.pt'.")

    elif uploaded_file is None:
        st.markdown("""
        <div style="
            display:flex; flex-direction:column; text-align:center;
            align-items:center; justify-content:center;
            height:330px; color:#999;">
            <i class="fa-solid fa-microscope" style="font-size:5rem; opacity:0.25;"></i>
            <p style="font-size:0.95rem;">Sube una imagen y presiona <b>Iniciar Análisis</b>.</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        if analizar:
            with st.spinner("Analizando tejido pulmonar..."):
                time.sleep(1.3)
                tensor = preprocess_image(image)
                clase, confianza, probs = predict_image(model, tensor)

            # Mostrar resultados
            st.markdown(f"<h2>{clase}</h2>", unsafe_allow_html=True)
            st.write("Confianza:", f"{confianza:.2f}%")
            st.progress(int(confianza)/100)

# ============================
#           FOOTER
# ============================

st.markdown("""
<div style="text-align: center; padding: 2rem; color: #666; font-size: 0.8rem;">
    © 2025 DeepMed AI Solutions. Solo para fines de investigación académica.
</div>
""", unsafe_allow_html=True)
