import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np


# ============================================
# 🔵 1. Arquitectura del Modelo CNN (tu modelo)
# ============================================

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



# =====================================================
# 🔵 2. Función para cargar el modelo entrenado (.pt)
# =====================================================

def load_model(model_path="modelo_cnn_completo.pt", device="cpu"):
    """
    Carga el modelo CNN entrenado desde un archivo .pt.
    """
    try:
        model = torch.load(model_path, map_location=torch.device(device))
        model.eval()
        return model
    except Exception as e:
        print("❌ ERROR al cargar el modelo:", e)
        return None



# =====================================================
# 🔵 3. Preprocesamiento de imagen para predicción
# =====================================================

def preprocess_image(image_file):
    """
    Recibe:
        - image_file: ruta o BytesIO (Streamlit / FastAPI / Flask)

    Devuelve:
        - tensor preprocesado (1, 3, 224, 224)
    """
    image = Image.open(image_file).convert("RGB")
    image = image.resize((224, 224))

    transform = transforms.Compose([
        transforms.ToTensor(),          # normaliza de 0 a 1
    ])

    tensor = transform(image).unsqueeze(0)  # (1, 3, 224, 224)
    return tensor



# =====================================================
# 🔵 4. Predicción final del modelo CNN
# =====================================================

def predict_image(model, tensor, device="cpu"):
    """
    Recibe:
        - model: Red cargada
        - tensor: imagen preprocesada (1,3,224,224)

    Devuelve:
        - clase_predicha (string)
        - confianza (%)
        - vector de probabilidades (numpy)
    """
    categorias = ["Bengin cases", "Malignant cases", "Normal cases"]

    tensor = tensor.to(device)

    with torch.no_grad():
        salida = model(tensor)
        probabilidades = F.softmax(salida, dim=1).cpu().numpy()[0]

    idx_pred = np.argmax(probabilidades)
    clase = categorias[idx_pred]
    confianza = probabilidades[idx_pred] * 100

    return clase, confianza, probabilidades
