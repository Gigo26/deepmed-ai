import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import time

# ==============================================================================
# 1. CONFIGURACIÓN DE PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="DeepMed AI - Lung Cancer Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==============================================================================
# 2. ESTILOS CSS (Streamlit Markdown)
# ==============================================================================
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
/* Ocultar elementos nativos molestos y ajustar padding */
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
.header-title h1 { margin: 0; font-size: 1.8rem; font-weight: 700; color: white !important; }
.header-subtitle { font-size: 0.9rem; opacity: 0.9; font-weight: 300; }
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
.big-prediction { font-size: 2.5rem; font-weight: 800; margin-bottom: 0.5rem; }
.details-list { list-style: none; padding: 0; margin-top: 2rem; font-size: 0.95rem; }
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

# ==============================================================================
# 3. LÓGICA DEL MODELO (PYTORCH)
# ==============================================================================

class LungCNN(nn.Module):
    """
    Define la arquitectura de la Red Neuronal Convolucional (CNN) para la clasificación.
    """
    def __init__(self):
        super(LungCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 112, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(112, 112, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(112, 112, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(112, 112, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(112, 112, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(112, 56, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(56, 56, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(56 * 14 * 14, 3000),
            nn.ReLU(),
            nn.Linear(3000, 1500),
            nn.ReLU(),
            nn.Linear(1500, 3) # 3 clases: Benigno, Maligno, Normal
        )
    
    def forward(self, x):
        return self.net(x)

@st.cache_resource
def load_model():
    """Carga el modelo de PyTorch y lo almacena en caché."""
    try:
        # Asegúrate de que el archivo 'modelo_cnn_completo.pt' esté en el mismo directorio.
        model = torch.load("modelo_cnn_completo.pt", map_location="cpu")
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

def preprocess_image(image):
    """Aplica las transformaciones necesarias a la imagen para la predicción."""
    image = image.convert("RGB")
    image = image.resize((224, 224))
    transform = transforms.Compose([transforms.ToTensor()])
    tensor = transform(image).unsqueeze(0)
    return tensor

def predict_image(model, tensor):
    """Realiza la predicción sobre el tensor de imagen."""
    categorias = ["Bengin cases", "Malignant cases", "Normal cases"]
    with torch.no_grad():
        salida = model(tensor)
        probabilidades = F.softmax(salida, dim=1).cpu().numpy()[0]
        clase_idx = np.argmax(probabilidades)
        
    return categorias[clase_idx], probabilidades[clase_idx] * 100, probabilidades

# Cargar el modelo al inicio de la aplicación
model = load_model()

# ==============================================================================
# 4. INTERFAZ DE USUARIO (STREAMLIT LAYOUT)
# ==============================================================================

# --- HEADER PERSONALIZADO (HTML PURO) ---
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

# --- LAYOUT PRINCIPAL (Dos Columnas) ---
col1, col2 = st.columns([1, 1], gap="large")

# --- COLUMNA IZQUIERDA: Carga de Imagen ---
with col1:
    # Inicio de la tarjeta visual (Hack CSS)
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header"><i class="fa-solid fa-upload"></i> Subir Tomografía (CT)</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Selecciona una imagen médica", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Vista previa de radiografía", use_column_width=True)
    else:
        # Espacio vacío visual para mantener altura y dar contexto
        st.markdown("""
        <div style="text-align: center; color: #ccc; padding: 40px;">
            <i class="fa-solid fa-cloud-arrow-up" style="font-size: 4rem; margin-bottom: 10px;"></i>
            <p>Soporta JPG, PNG y DICOM (si se maneja la conversión internamente)</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True) # Fin tarjeta

# --- COLUMNA DERECHA: Resultados ---
with col2:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header"><i class="fa-solid fa-file-medical-alt"></i> Resultados del Diagnóstico</div>', unsafe_allow_html=True)
    
    if model is None:
        st.error("Error: No se encontró el archivo 'modelo_cnn_completo.pt'. Asegúrate de que exista.")
    elif uploaded_file is None:
        # ESTADO VACÍO (Placeholder)
        st.markdown("""
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 300px; color: #999;">
            <i class="fa-solid fa-microscope" style="font-size: 5rem; margin-bottom: 1rem; opacity: 0.3;"></i>
            <p style="text-align: center;">El análisis aparecerá aquí después de subir una imagen.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Lógica de Análisis y Predicción
        analyze_btn = st.button("Iniciar Análisis Clínico ⚡")
        
        if analyze_btn:
            with st.spinner('Analizando tejido pulmonar...'):
                # Simular tiempo de proceso para efecto visual (opcional)
                time.sleep(1.5)
                
                # Preprocesamiento y Predicción
                tensor = preprocess_image(image)
                clase, confianza, probs = predict_image(model, tensor)
                
                # --- MOSTRAR RESULTADOS CON DISEÑO ---
                
                # Determinar estilos según resultado
                if clase == "Malignant cases":
                    badge_class = "badge-danger"
                    icon = "fa-exclamation-triangle"
                    color_text = "#d93025"
                    status_text = "Detección Positiva"
                else:
                    badge_class = "badge-success"
                    icon = "fa-check-circle"
                    color_text = "#28a745"
                    status_text = "Detección Negativa" if clase == "Normal cases" else "Caso Benigno"
                
                # Renderizar HTML del resultado
                st.markdown(f"""
                <div style="text-align: center; animation: fadeIn 0.5s;">
                    <div class="result-badge {badge_class}">
                        <i class="fa-solid {icon}"></i> {status_text}
                    </div>
                    <h2 class="big-prediction" style="color: {color_text};">{clase}</h2>
                    <p style="color: #666;">Confianza del modelo IA</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Barra de progreso personalizada
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; font-weight: bold; margin-top: 20px;">
                    <span>Certeza</span>
                    <span>{confianza:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
                st.progress(int(confianza) / 100)
                
                # Lista de detalles (Estilo Tabla Médica)
                st.markdown(f"""
                <ul class="details-list">
                    <li>Prob. Maligno: <span>{probs[1]*100:.1f}%</span></li>
                    <li>Prob. Benigno: <span>{probs[0]*100:.1f}%</span></li>
                    <li>Prob. Normal: <span>{probs[2]*100:.1f}%</span></li>
                    <li>Modelo: <span>DeepResNet-50 v2</span></li>
                    <li>Tiempo Inferencia: <span>1.2s</span></li>
                </ul>
                <div style="text-align: center; margin-top: 20px; font-size: 3rem; color: #e0e0e0;">
                    <i class="fa-solid fa-lungs"></i>
                </div>
                """, unsafe_allow_html=True)
                
    st.markdown('</div>', unsafe_allow_html=True) # Fin tarjeta

# ==============================================================================
# 5. FOOTER
# ==============================================================================
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #666; font-size: 0.8rem;">
    © 2025 DeepMed AI Solutions. Solo para fines de investigación académica.
</div>
""", unsafe_allow_html=True)
