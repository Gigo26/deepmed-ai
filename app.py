import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import numpy as np
import time

# ==========================================================
#  MODELO RESNET50 (para compatibilidad al cargar .pt)
# ==========================================================
class LungResNet50(nn.Module):
    def __init__(self):
        super(LungResNet50, self).__init__()

        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        for param in self.model.parameters():
            param.requires_grad = False

        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 3)
        )

    def forward(self, x):
        return self.model(x)

# ==========================================================
# 1. MODELO CNN (SE DEFINE PRIMERO)
# ==========================================================
class LungCNN(nn.Module):
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
            nn.Linear(1500, 3)
        )

    def forward(self, x):
        return self.net(x)

# ==========================================================
# 1. MODELO EFFICIENTNET
# ==========================================================
class LungEfficientNet(nn.Module):
    def __init__(self):
        super(LungEfficientNet, self).__init__()

        # Cargar EfficientNetB0 preentrenada
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Congelar características
        for param in self.model.features.parameters():
            param.requires_grad = False

        # Reemplazar la capa final
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, 3)

    def forward(self, x):
        return self.model(x)

# ==========================================================
# 1. MODELO VGG16
# ==========================================================
class LungVGG16(nn.Module):
    def __init__(self):
        super(LungVGG16, self).__init__()

        # Cargar modelo base con pesos ImageNet
        self.vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        # Congelar capas convolucionales
        for param in self.vgg.features.parameters():
            param.requires_grad = False

        # Reemplazar clasificador
        self.vgg.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(25088, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 3)     # 3 clases
        )

    def forward(self, x):
        return self.vgg(x)

# ==========================================================
# 2. TRANSFORMACIONES Y CLASES
# ==========================================================
# -------------------------
# 1) Transform para tu CNN
# -------------------------
transform_cnn = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# -------------------------
# 2) Transform para VGG16 y ResNet50
# -------------------------
transform_vgg = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------
# 3) Transform para EfficientNetB0
# -------------------------
transform_eff = EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()

CLASSES = ["Benigno", "Maligno", "Normal"]

# ==========================================================
# 3. CARGAR MODELO ENTRENADO CNN
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ruta_modelo = "modelo_cnn_completo.pt"

# Cargar el modelo completo (arquitectura + pesos)
model = torch.load(ruta_modelo, map_location=device)
model.to(device)
model.eval()

# ==========================================================
# 3. CARGAR MODELO ENTRENADO RESNET50
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet_model = LungResNet50()
resnet_model.load_state_dict(torch.load("modelo_resnet50_modelo.pt", map_location=device))
resnet_model.eval()
resnet_model.to(device)

# ==========================================================
# 3. CARGAR MODELO ENTRENADO VGG16
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ruta_modelo_vgg = "modelo_vgg16_completo.pt"

vgg_model = torch.load(ruta_modelo_vgg, map_location=device)
vgg_model.eval()
vgg_model.to(device)

# ==========================================================
# 3. CARGAR MODELO ENTRENADO EFFICIENTNET
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ruta_modelo = "modelo_efficientnet_completo.pt"

# Cargar el modelo completo (arquitectura + pesos)
eff_model = torch.load("modelo_efficientnet_completo.pt", map_location=device)
eff_model.eval()
eff_model.to(device)
 
# ==========================================================
# 4. CONFIGURACIÓN DE PÁGINA
# ==========================================================
st.set_page_config(
    page_title="DeepMed AI",
    page_icon="🫁",
    layout="wide"
)

# Cargar Google Fonts + Icons
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
""", unsafe_allow_html=True)

# ==========================================================
# 5. CSS DEL HEADER
# ==========================================================
st.markdown("""
<style>
/* Ocultar header nativo de Streamlit */
[data-testid="stHeader"] {
    display: none !important;
}

/* HEADER FULL WIDTH PEGADO ARRIBA */
.custom-header {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    padding: 18px 32px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: linear-gradient(90deg, #00007A 0%, #6B6BDF 100%);
    color: white;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    z-index: 9999;
    font-family: 'Inter', sans-serif;
}

/* Ajuste correcto del contenido para evitar que quede detrás del header fijo */
.stMainBlockContainer {
    padding-top: 110px !important;
}

/* Layout del lado izquierdo */
.header-left {
    display: flex;
    align-items: center;
    gap: 20px;
}

/* Títulos */
.header-title {
    display: flex;
    flex-direction: column;
    gap: 2px;
}

.header-title-main {
    margin: 0;
    font-size: 28px;
    font-weight: 900;
    text-transform: uppercase;
    color: white;
    letter-spacing: 1.3px;
    line-height: 1;
}

.header-subtitle {
    margin: 0;
    font-size: 13px;
    font-weight: 300;
    opacity: 0.95;
    color: #e5e5e5;
    letter-spacing: 0.5px;
}

/* Íconos */
.icon-style {
    font-size: 34px;
    color: white;
}

/* Espaciador */
.header-spacer {
    flex-grow: 1;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# 6. CSS GLOBAL + TRUCO DEL FILE UPLOADER 100% FUNCIONAL
# ==========================================================
st.markdown("""
<style>
/* Fondo general */
[data-testid="stAppViewContainer"] {
    background-color: #E8F4F8;
    font-family: 'Inter', sans-serif;
}

/* ============================================================
   ESTILO FINAL DEL UPLOADER: NUBE ARRIBA, TEXTOS CENTRADOS
   ============================================================ */

/* 1. CONTENEDOR PRINCIPAL (Caja Punteada) */
[data-testid="stFileUploaderDropzone"] {
    border: 3px dashed #2C74B3;
    background-color: #D4E8F0;
    border-radius: 20px;
    padding: 40px;
    min-height: 350px; 
    
    /* OBLIGATORIO: Flexbox Vertical */
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    align-items: center !important;
}

/* Hover effect */
[data-testid="stFileUploaderDropzone"]:hover {
    background-color: #C5E0EB;
    border-color: #1E5A96;
}

/* 2. NUBE GIGANTE CON FLECHA (Insertada en el contenedor padre) */
[data-testid="stFileUploaderDropzone"]::before {
    content: "\\f0ee";  /* Código Unicode de fa-cloud-arrow-up */
    font-family: "Font Awesome 6 Free"; /* Nombre exacto de la fuente */
    font-weight: 900; /* Peso Bold obligatorio para íconos sólidos */
    
    font-size: 90px;
    color: #2C74B3;
    display: block;
    margin-bottom: 25px; /* Espacio entre nube y texto */
    line-height: 1;
}

/* 3. TEXTO PRINCIPAL: "Arrastra y suelta..." */
/* Reemplazamos el texto 'Drag and drop' usando ::after del div interno */
/* Primero ocultamos todo lo nativo */
[data-testid="stFileUploaderDropzone"] > div {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
}

[data-testid="stFileUploaderDropzone"] svg, 
[data-testid="stFileUploaderDropzone"] small, 
[data-testid="stFileUploaderDropzone"] span {
    display: none !important;
}

/* Inyectamos nuestro Título */
[data-testid="stFileUploaderDropzone"] > div::before {
    content: "Arrastra y suelta o agrega tu imagen aquí";
    font-size: 26px;
    font-weight: 900;
    color: #0A2647;
    margin-bottom: 10px;
    text-align: center;
}

/* Inyectamos nuestro Subtítulo */
[data-testid="stFileUploaderDropzone"] > div::after {
    content: "Soporta JPG, JPEG, PNG";
    font-size: 16px;
    color: #666;
    margin-bottom: 20px;
    text-align: center;
    display: block;
}

/* 4. BOTÓN "Seleccionar Archivo" */
[data-testid="stFileUploaderDropzone"] button {
    border: 2px solid #2C74B3;
    background-color: white;
    padding: 12px 30px;
    border-radius: 10px;
    font-weight: 700;
    font-size: 16px;
    color: transparent; /* Ocultar texto original */
    position: relative;
    transition: 0.3s;
    min-width: 220px;
    margin-top: 10px; /* Separar del subtítulo */
}

/* Texto nuevo del botón */
[data-testid="stFileUploaderDropzone"] button::after {
    content: "Seleccionar Archivo";
    position: absolute;
    color: #2C74B3;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    width: 100%;
}

[data-testid="stFileUploaderDropzone"] button:hover {
    background-color: #2C74B3;
}

[data-testid="stFileUploaderDropzone"] button:hover::after {
    color: white;
}

/* 1. Estado Normal del botón de Análisis */
div.stButton > button {
    background: linear-gradient(90deg, #7BA3C8 0%, #5B738A 100%) !important;
    color: white !important;
    border: none !important;
    height: 60px !important;
    font-size: 26px !important;
    font-weight: 900 !important;
    border-radius: 70px !important;
    box-shadow: 0 4px 10px rgba(0,0,0,0.2) !important;
    transition: transform 0.2s ease !important;
}

/* 2. Estado Hover (Mouse encima) */
div.stButton > button:hover {
    background: linear-gradient(90deg, #5B738A 0%, #7BA3C8 100%) !important; /* Invertir degradado */
    color: white !important;
    transform: scale(1.02) !important; /* Pequeño efecto zoom */
    box-shadow: 0 6px 12px rgba(0,0,0,0.3) !important;
}

/* 3. Estado Active/Focus (Cuando haces clic) - QUITA EL BORDE ROJO DE STREAMLIT */
div.stButton > button:active, 
div.stButton > button:focus {
    color: white !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}

/* 4. Asegurar que el texto dentro del botón sea visible */
div.stButton > button p {
    font-size: 26px !important; 
    font-weight: 900 !important;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# 7. DISEÑO PARA BODY - AJUSTES DE ESPACIOS
# ==========================================================
st.markdown("""
<style>
/* Reducimos el padding general del body */
.block-container {
    padding-top: 40px !important;
    padding-bottom: 20px !important;
}

/* Ajustamos espacio después del header */
.stMainBlockContainer {
    padding-top: 80px !important;
}

/* Reducir alto del uploader */
[data-testid="stFileUploaderDropzone"] {
    min-height: 260px !important;
    padding: 25px !important; 
}

/* Reducir tamaño de textos del uploader */
[data-testid="stFileUploaderDropzone"] > div::before {
    font-size: 22px !important;
    margin-bottom: 6px !important;
}
[data-testid="stFileUploaderDropzone"] > div::after {
    font-size: 14px !important;
}

/* Reducir tamaño de la nube */
[data-testid="stFileUploaderDropzone"]::before {
    font-size: 65px !important;
    margin-bottom: 18px !important;
}

/* Compactar columnas (reduce el aire vertical) */
.css-1y4p8pa {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

/* Compactar el título principal */
h2 {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

/* Reducir espacios entre elementos en general */
.element-container {
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# 8. HEADER HTML (SECCIÓN SUPERIOR)
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
    <div class="header-spacer"></div>
    <i class="fa-solid fa-user-md icon-style" title="Medical Staff"></i>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# 9. LAYOUT DE DOS COLUMNAS 
# ==========================================================
col1, col2 = st.columns([1, 1], gap="large")

# ==========================================================
# COLUMNA 1 — SUBIR IMAGEN
# ==========================================================
with col1:

    st.markdown("""
        <h2 style="font-weight:900; color:#0A2647;">
            <i class="fa-solid fa-cloud-upload-alt"></i> Sube tu Imagen Radiológica
        </h2>
    """, unsafe_allow_html=True)

    # --- Subidor de archivo ---
    uploaded_file = st.file_uploader(
        "Carga la imagen de rayos X",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        label_visibility="collapsed"
    )

    # Mostrar vista previa
    if uploaded_file:
        st.image(uploaded_file, caption="Imagen subida", use_container_width=True)

    # Botón de análisis
    analyze_clicked = st.button("Iniciar Análisis", use_container_width=True)

    # ===============================
    #    🔥 EJECUTAR ANÁLISIS
    # ===============================
    if analyze_clicked:
        if uploaded_file is None:
            st.error("⚠️ Por favor sube una imagen primero")
        else:
            try:
                # --- Procesar la imagen una sola vez ---
                image = Image.open(uploaded_file).convert("RGB")
                # 1. CNN (sin normalización ImageNet)
                img_cnn = transform_cnn(image).unsqueeze(0).to(device)
                
                # 2. VGG16 (normalización ImageNet)
                img_vgg = transform_vgg(image).unsqueeze(0).to(device)
                
                # 3. ResNet50 (igual que VGG: normalización ImageNet)
                img_res = transform_vgg(image).unsqueeze(0).to(device)
                
                # 4. EfficientNet (normalización especial)
                transform_eff = EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()
                img_eff = transform_eff(image).unsqueeze(0).to(device)

                # =====================
                #    🔹 1) CNN
                # =====================
                start_cnn = time.time()
                with torch.no_grad():
                    out_cnn = model(img_cnn)
                    probs_cnn = torch.softmax(out_cnn, dim=1)
                    conf_cnn, pred_cnn = torch.max(probs_cnn, 1)

                cnn_diag = CLASSES[pred_cnn.item()]
                cnn_conf = float(conf_cnn.item() * 100)
                cnn_time = round(time.time() - start_cnn, 3)

                # =============================
                #    🔹 2) EfficientNetB0
                # =============================
                start_eff = time.time()
                with torch.no_grad():
                    out_eff = eff_model(img_eff)
                    probs_eff = torch.softmax(out_eff, dim=1)
                    conf_eff, pred_eff = torch.max(probs_eff, 1)

                eff_diag = CLASSES[pred_eff.item()]
                eff_conf = float(conf_eff.item() * 100)
                eff_time = round(time.time() - start_eff, 3)

                # =============================
                #     3) VGG16 
                # =============================
                start_vgg = time.time()
                with torch.no_grad():
                    out_vgg = vgg_model(img_vgg)
                    probs_vgg = torch.softmax(out_vgg, dim=1)
                    conf_vgg, pred_vgg = torch.max(probs_vgg, 1)
                
                vgg_diag = CLASSES[pred_vgg.item()]
                vgg_conf = float(conf_vgg.item() * 100)
                vgg_time = round(time.time() - start_vgg, 3)

                # =============================
                #     4) RESNET50
                # =============================
                start_res = time.time()
                with torch.no_grad():
                    out_res = resnet_model(img_res)
                    probs_res = torch.softmax(out_res, dim=1)
                    conf_res, pred_res = torch.max(probs_res, 1)
                
                res_diag = CLASSES[pred_res.item()]
                res_conf = float(conf_res.item() * 100)
                res_time = round(time.time() - start_res, 3)
                
                # =============================
                #    🔥 GUARDAR RESULTADOS
                # =============================
                st.session_state["multi_results"] = {
                    "CNN":            (cnn_diag, cnn_conf, cnn_time),
                    "EfficientNetB0": (eff_diag, eff_conf, eff_time),
                    "VGG16":            (vgg_diag, vgg_conf, vgg_time),
                    "ResNet50":       (res_diag, res_conf, res_time)
                }

                st.session_state["analysis_complete"] = True

                st.rerun()

            except Exception as e:
                st.error(f"Error durante el análisis: {e}")
# ==========================================================
# COLUMNA 2 — RESULTADOS
# ==========================================================
with col2:

    st.markdown("""
        <h2 style="font-weight:900; color:#0A2647;">
            <i class="fa-solid fa-file-medical-alt"></i> Resultados del Diagnóstico
        </h2>
        <hr>
    """, unsafe_allow_html=True)

    # ------------------------------------------------------
    #   FUNCIÓN DE COLOR POR DIAGNÓSTICO
    # ------------------------------------------------------
    def get_diag_color(diag):
        colores = {
            "Normal": "#28A745",   # Verde
            "Benigno": "#FFC107",  # Amarillo
            "Maligno": "#DC3545"   # Rojo
        }
        return colores.get(diag, "#5B6DFF")  # Azul si no coincide

    # ------------------------------------------------------
    #   SI HAY RESULTADOS → MOSTRAR TABLA
    # ------------------------------------------------------
    if "analysis_complete" in st.session_state:

        results = st.session_state["multi_results"]

        st.markdown("""
        <h2 style='font-weight:900; color:#0A2647; margin-bottom:10px;'>
            <i class="fa-solid fa-chart-column"></i> Comparación de Modelos
        </h2>
        <hr>
        """, unsafe_allow_html=True)

        # 💡 OJO: todas las líneas HTML empiezan con "<", sin espacios
        tabla_html = (
            "<table style='width:100%; border-collapse:collapse; "
            "font-family:Inter; font-size:16px; "
            "border-radius:10px; overflow:hidden;'>"
            "<tr style='background:#0A2647; color:white; text-align:center;'>"
            "<th style='padding:10px;'>Modelo</th>"
            "<th style='padding:10px;'>Predicción</th>"
            "<th style='padding:10px;'>Confianza</th>"
            "<th style='padding:10px;'>Tiempo</th>"
            "</tr>"
        )

        for modelo, (diag, conf, tiempo) in results.items():
            bg = get_diag_color(diag)
            bg_soft = bg + "20"

            tabla_html += (
                f"<tr style='text-align:center; background:{bg_soft};'>"
                f"<td style='padding:10px; font-weight:700; color:#0A2647;'>{modelo}</td>"
                f"<td style='padding:10px; color:{bg}; font-weight:900;'>{diag}</td>"
                f"<td style='padding:10px; font-weight:900; color:#0A2647;'>{conf:.1f}%</td>"
                f"<td style='padding:10px; color:#0A2647;'>{tiempo}s</td>"
                "</tr>"
            )

        tabla_html += "</table>"

        st.markdown(tabla_html, unsafe_allow_html=True)

    else:
        # Mensaje inicial si no hay análisis
        st.markdown("""
            <div style="text-align:center; padding:20px;">
                <i class="fa-solid fa-microscope" 
                style="font-size:60px; color:#0A2647; margin-bottom:15px;"></i>
                <p style="color:#777; font-size:15px;">
                    Sube una imagen y presiona <b>“Iniciar Análisis”</b>.
                </p>
            </div>
        """, unsafe_allow_html=True)
