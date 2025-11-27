import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

# ============================
#     CONFIGURACIÓN GENERAL
# ============================

st.set_page_config(
    page_title="Detección de Cáncer de Pulmón - DeepMed AI",
    layout="wide"
)

# ============================
#           ESTILOS CSS
# ============================
st.markdown("""
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

.block-container {
    padding-top: 1.5rem !important;
}

.card {
    background: var(--white);
    padding: 1.8rem;
    border-radius: var(--radius);
    box-shadow: 0 10px 30px rgba(0,0,0,0.05);
}

.upload-box {
    border: 2px dashed var(--accent-blue);
    border-radius: var(--radius);
    padding: 2rem;
    background-color: #f8fbff;
    text-align: center;
}

.result-good {
    color: var(--success-green);
    font-weight: 700;
    font-size: 1.8rem;
}

.result-bad {
    color: red;
    font-weight: 700;
    font-size: 1.8rem;
}

.conf-bar {
    height: 15px;
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)



# ============================
#    CLASE DEL MODELO CNN
# ============================
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



# ============================
#   CARGAR EL MODELO GUARDADO
# ============================
@st.cache_resource
def load_model():
    model = torch.load("modelo_cnn_completo.pt", map_location="cpu")
    model.eval()
    return model

model = load_model()


# ============================
#     PREPROCESAR IMAGEN
# ============================
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    tensor = transform(image).unsqueeze(0)
    return tensor


# ============================
#        PREDICCIÓN
# ============================
def predict_image(model, tensor):
    categorias = ["Bengin cases", "Malignant cases", "Normal cases"]

    with torch.no_grad():
        salida = model(tensor)
        probabilidades = F.softmax(salida, dim=1).cpu().numpy()[0]

    clase = np.argmax(probabilidades)
    confianza = probabilidades[clase] * 100

    return categorias[clase], confianza, probabilidades



# ============================
#        INTERFAZ STREAMLIT
# ============================

st.title("🫁 DeepMed AI — Lung Cancer Detection System")
st.write("### Inteligencia Artificial para la detección temprana de nódulos pulmonares")

col1, col2 = st.columns([1, 1])

# -------- SUBIR IMAGEN ----------
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📤 Subir Tomografía (CT)")

    file = st.file_uploader(
        "Arrastra o selecciona una imagen (JPG, PNG)",
        type=["jpg", "jpeg", "png"]
    )

    image = None
    if file:
        image = Image.open(file)
        st.image(image, caption="Imagen cargada", use_column_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


# -------- RESULTADOS ----------
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📋 Resultados del Diagnóstico")

    if file is None:
        st.info("Sube una imagen y espera los resultados.")
    else:
        tensor = preprocess_image(image)
        clase, confianza, probs = predict_image(model, tensor)

        st.markdown("### Resultado principal:")

        if clase == "Malignant cases":
            st.markdown(f'<p class="result-bad">🛑 {clase}</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="result-good">🟢 {clase}</p>', unsafe_allow_html=True)

        st.write(f"**Nivel de confianza:** {confianza:.2f}%")

        st.progress(int(confianza))

        st.write("### Probabilidades por clase:")
        st.write(f"- **Bengin:** {probs[0]*100:.2f}%")
        st.write(f"- **Malignant:** {probs[1]*100:.2f}%")
        st.write(f"- **Normal:** {probs[2]*100:.2f}%")

    st.markdown("</div>", unsafe_allow_html=True)

