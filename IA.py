from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np

# Cargamos la red ya entrenada
model = load_model("cnn.keras")

def predict(test_image):
    # Cargar y preprocesar la imagen para que coincida con el tamaño esperado por el modelo
    classes = ["ajedrez", "baloncesto", "boxeo", "disparo", "esgrima", "formula1",
                "futbol", "hockey", "natacion", "tenis"]
    
    # Predecir la clase
    result = model.predict(test_image)
    
    # Mostrar los valores predichos (probabilidades para cada clase)
    print("Probabilidades predichas:", result)
    
    # Obtener el índice de la clase con mayor probabilidad
    predicted_class_index = np.argmax(result)
    
    # Obtener el nombre de la clase predicha
    predicted_class_label = classes[predicted_class_index]
    
    # Imprimir la clase predicha
    print(f'La imagen pertenece a la clase: {predicted_class_label}')

    return predicted_class_label
