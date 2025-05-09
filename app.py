from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from flask import Flask, request, jsonify
from flask_cors import CORS  # <-- Asegúrate de tener esto instalado
import IA
import os

app = Flask(__name__)  # <-- ¡Los dobles guiones bajos en __name__!

# Configuración CORRECTA de CORS (sin errores de paréntesis)
CORS(app, resources={r"/*": {"origins": "*"}})

app.config['UPLOAD_FOLDER'] = './uploads'  # Ahora línea 21 debería funcionar
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'gif'}
    
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No se recibió la imagen"}), 400

    file = request.files['image']

    # 1. Cargar la imagen y redimensionar
    img = Image.open(file.stream).convert('RGB').resize((64, 64))

    # 2. Convertir a array y normalizar (como durante el entrenamiento)
    img_array = np.array(img)  # Shape: (64, 64, 3)
    img_array = img_array.astype('float32') / 255.0  # Normalizar a [0, 1]

    # 3. Añadir dimensión de batch (shape: (1, 64, 64, 3))
    img_array = np.expand_dims(img_array, axis=0)

    # 4. Debug: Verificar shape y normalización
    print("Shape:", img_array.shape)  # Debe ser (1, 64, 64, 3)
    print("Valores mín/máx:", img_array.min(), img_array.max())  # Debe ser 0.0 y 1.0

    # 5. Predecir
    resultado_ia = IA.predict(img_array)
    return jsonify({"resultado": resultado_ia})

if __name__ == "__main__":
  port = int(os.environ.get("PORT", 5000))
  app.run(host="0.0.0.0", port=port)

