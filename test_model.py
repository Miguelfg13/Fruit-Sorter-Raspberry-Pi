import cv2
import numpy as np
import tensorflow.lite as tflite
import os
import sys
import random

# ================= CONFIGURACIÓN =================
TFLITE_MODEL_PATH = 'fruit_model.tflite'
IMG_SIZE = 224
CLASSES = ['cereza_buena', 'cereza_mala', 'fresa_buena', 'fresa_mala']
VAL_DIR = os.path.join('dataset', 'validation')

# ================= FUNCIONES (Copiadas del script de Pi) =================

def preprocesar_imagen(frame):
    """Prepara la imagen capturada para el modelo."""
    # Convertir de BGR (OpenCV) a RGB (Modelo)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Redimensionar a lo que espera el modelo
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    
    # Expandir dimensiones (batch_size = 1)
    img_input = np.expand_dims(img_resized, axis=0)
    
    # Convertir a Float32 (el modelo espera 0-255 en float)
    # NOTA: NO normalizamos a 0-1 ni -1 a 1 aquí, 
    # porque la capa preprocess_input ya está DENTRO del .tflite
    img_input = img_input.astype(np.float32)
    
    return img_input

def clasificar(interpreter, img_input):
    """Ejecuta la inferencia en un frame de imagen."""
    
    # Obtener detalles de entrada y salida
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # 2. Asignar imagen al tensor de entrada
    interpreter.set_tensor(input_details[0]['index'], img_input)
    
    # 3. Ejecutar inferencia
    interpreter.invoke()
    
    # 4. Obtener resultados
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = output_data[0]
    
    # 5. Interpretar resultados
    predicted_index = np.argmax(prediction)
    clase_predicha = CLASSES[predicted_index]
    confianza = float(prediction[predicted_index])
    
    return clase_predicha, confianza, prediction

def get_random_image_path():
    """Elige una imagen aleatoria de la carpeta de validación."""
    try:
        random_class = random.choice(CLASSES)
        class_path = os.path.join(VAL_DIR, random_class)
        random_image_name = random.choice(os.listdir(class_path))
        full_path = os.path.join(class_path, random_image_name)
        return full_path
    except Exception as e:
        print(f"Error al buscar imagen aleatoria: {e}")
        print("Asegúrate que la carpeta 'dataset/validation' exista y tenga imágenes.")
        return None

# ==================== MAIN ====================

def main():
    # --- 1. Cargar modelo TFLite ---
    try:
        interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)
        interpreter.allocate_tensors()
        print(f"✅ Modelo {TFLITE_MODEL_PATH} cargado exitosamente.")
    except ValueError:
        print(f"❌ Error: No se pudo cargar {TFLITE_MODEL_PATH}.")
        print("Asegúrate de que 'tensorflow' o 'tflite-runtime' esté instalado.")
        return
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        return

    # --- 2. Obtener ruta de la imagen ---
    if len(sys.argv) > 1:
        # Si el usuario proporciona una ruta
        image_path = sys.argv[1]
        print(f"Iniciando prueba con imagen: {image_path}")
    else:
        # Si no, tomar una aleatoria
        print("No se proporcionó imagen. Buscando una aleatoria en 'dataset/validation'...")
        image_path = get_random_image_path()
        if not image_path:
            return
        print(f"Probando con imagen aleatoria: {image_path}")

    # --- 3. Cargar y procesar la imagen ---
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ Error: No se pudo leer la imagen en {image_path}")
        return

    # Copia para visualización
    frame_display = frame.copy()
    
    # Preprocesar
    img_input = preprocesar_imagen(frame)

    # --- 4. Clasificar ---
    print("\nEjecutando inferencia...")
    clase, conf, raw_predictions = clasificar(interpreter, img_input)

    # --- 5. Mostrar resultados ---
    print("\n" + "="*30)
    print("         RESULTADO DE LA PRUEBA")
    print("="*30)
    print(f"➡️ Predicción:   {clase.upper()}")
    print(f"Confidence:   {conf * 100:.2f}%")
    print("="*30)
    
    print("\nDetalle de predicciones:")
    for i, class_name in enumerate(CLASSES):
        print(f"  - {class_name:<15}: {raw_predictions[i] * 100:.2f}%")
        
    # --- 6. Visualizar imagen ---
    # Poner texto en la imagen
    text = f"{clase} ({conf*100:.1f}%)"
    color = (0, 255, 0) # Verde
    
    # Redimensionar para mostrar si es muy grande
    max_h, max_w = 600, 800
    h, w = frame_display.shape[:2]
    scale = min(max_w/w, max_h/h)
    if scale < 1:
        h_new, w_new = int(h*scale), int(w*scale)
        frame_display = cv2.resize(frame_display, (w_new, h_new))
    
    cv2.putText(frame_display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    print("\nMostrando imagen. Presiona cualquier tecla para salir.")
    cv2.imshow("Prueba de Modelo TFLite", frame_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()