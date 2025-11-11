#!/usr/bin/env python3
"""
Script de Inferencia y Control para Raspberry Pi
Controla una banda transportadora y servos para clasificar frutas
usando un modelo TFLite entrenado.
"""

import RPi.GPIO as GPIO
import time
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from threading import Thread

# ==================== CONFIGURACIÓN DE HARDWARE ====================

# --- Pines (Modo BCM) ---
# Asegúrate de conectar tus componentes a estos pines GPIO
PIN_SENSOR_CAMARA = 17 # Pin de entrada para el sensor de proximidad
PIN_RELE_BANDA = 27    # Pin de salida para el relé que controla la banda
PIN_SERVO_CEREZA = 18  # Pin PWM para el servo de cerezas buenas
PIN_SERVO_FRESA = 22   # Pin PWM para el servo de fresas buenas

# --- Lógica de Hardware ---
# Ajusta esto según cómo funcionen tus sensores/relés
SENSOR_DETECTA_EN = GPIO.LOW  # ¿El sensor detecta en LOW (0V) o HIGH (3.3V)?
RELE_BANDA_ON = GPIO.HIGH     # ¿El relé enciende la banda con HIGH o LOW?

# --- Configuración de Servos ---
SERVO_HZ = 50                 # Frecuencia estándar para servos
SERVO_ANGULO_REPOSO = 0
SERVO_ANGULO_ACTIVO = 45      # Grados que se moverá el servo
TIEMPO_SERVO_ACTIVO = 0.5     # Segundos que el servo espera antes de regresar

# ================= CONFIGURACIÓN DEL MODELO IA =================
TFLITE_MODEL_PATH = 'fruit_model.tflite' # ¡Asegúrate que este archivo esté en la misma carpeta!
IMG_SIZE = 224
CLASSES = ['cereza_buena', 'cereza_mala', 'fresa_buena', 'fresa_mala']

# Variables globales para los objetos de hardware
servo_cereza_pwm = None
servo_fresa_pwm = None
interpreter = None
input_details = None
output_details = None
cam = None

# ==================== FUNCIONES DE HARDWARE ====================

def mover_servo(pwm_obj, angulo):
    """Mueve un servo a un ángulo específico."""
    # Fórmula estándar: 50Hz -> 2% a 12% de ciclo de trabajo (0-180 grados)
    duty_cycle = (angulo / 18) + 2
    pwm_obj.ChangeDutyCycle(duty_cycle)

def accionar_clasificador(clase):
    """Mueve el servo correcto basado en la clasificación."""
    print(f"Accionando para: {clase}")
    
    if clase == 'cereza_buena':
        # Mover servo de cereza
        mover_servo(servo_cereza_pwm, SERVO_ANGULO_ACTIVO)
        time.sleep(TIEMPO_SERVO_ACTIVO)
        mover_servo(servo_cereza_pwm, SERVO_ANGULO_REPOSO)
        
    elif clase == 'fresa_buena':
        # Mover servo de fresa
        mover_servo(servo_fresa_pwm, SERVO_ANGULO_ACTIVO)
        time.sleep(TIEMPO_SERVO_ACTIVO)
        mover_servo(servo_fresa_pwm, SERVO_ANGULO_REPOSO)
        
    else:
        # 'cereza_mala' o 'fresa_mala' -> No hacer nada, dejar pasar
        print("Producto malo. Dejando pasar.")
        pass
    
    # Damos un pequeño tiempo para que el producto caiga/pase
    time.sleep(0.3)

def controlar_banda(arrancar):
    """Enciende o detiene el relé de la banda transportadora."""
    if arrancar:
        print("...Banda REANUDADA")
        GPIO.output(PIN_RELE_BANDA, RELE_BANDA_ON)
    else:
        print("...Banda DETENIDA")
        # El opuesto a RELE_BANDA_ON
        estado_off = GPIO.LOW if RELE_BANDA_ON == GPIO.HIGH else GPIO.HIGH
        GPIO.output(PIN_RELE_BANDA, estado_off)

def setup_hardware():
    """Configura todos los pines GPIO."""
    global servo_cereza_pwm, servo_fresa_pwm
    
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    
    # Configurar Sensor (Entrada)
    # Usamos PULL_UP si el sensor detecta en LOW (conecta a GND al detectar)
    if SENSOR_DETECTA_EN == GPIO.LOW:
        GPIO.setup(PIN_SENSOR_CAMARA, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    else:
        GPIO.setup(PIN_SENSOR_CAMARA, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        
    # Configurar Relé (Salida)
    GPIO.setup(PIN_RELE_BANDA, GPIO.OUT)
    
    # Configurar Servos (Salida PWM)
    GPIO.setup(PIN_SERVO_CEREZA, GPIO.OUT)
    GPIO.setup(PIN_SERVO_FRESA, GPIO.OUT)
    
    servo_cereza_pwm = GPIO.PWM(PIN_SERVO_CEREZA, SERVO_HZ)
    servo_fresa_pwm = GPIO.PWM(PIN_SERVO_FRESA, SERVO_HZ)
    
    # Iniciar servos en 0% (inactivos) y luego mover a reposo
    servo_cereza_pwm.start(0)
    servo_fresa_pwm.start(0)
    mover_servo(servo_cereza_pwm, SERVO_ANGULO_REPOSO)
    mover_servo(servo_fresa_pwm, SERVO_ANGULO_REPOSO)
    print("Servos en posición de reposo.")

# ================= FUNCIONES DE IA (VISIÓN) =================

def cargar_modelo():
    """Carga el modelo TFLite y prepara el intérprete."""
    global interpreter, input_details, output_details
    
    interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Verificar si el modelo espera float o int
    print(f"Modelo cargado. Tipo de entrada: {input_details[0]['dtype']}")

def preprocesar_imagen(frame):
    """Prepara la imagen capturada para el modelo."""
    # Convertir de BGR (OpenCV) a RGB (Modelo)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Redimensionar a lo que espera el modelo
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    
    # Expandir dimensiones (batch_size = 1)
    img_input = np.expand_dims(img_resized, axis=0)
    
    # Convertir a Float32 (el modelo espera 0-255 en float)
    img_input = img_input.astype(np.float32)
    
    return img_input

def clasificar(frame):
    """Ejecuta la inferencia en un frame de imagen."""
    
    # 1. Preprocesar la imagen
    img_input = preprocesar_imagen(frame)
    
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
    
    return clase_predicha, confianza

# ==================== BUCLE PRINCIPAL ====================

def main():
    global cam
    try:
        # --- Inicialización ---
        print("Iniciando sistema clasificador...")
        setup_hardware()
        cargar_modelo()
        
        # Iniciar cámara
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            print("❌ ERROR: No se pudo abrir la cámara.")
            return
        print("Cámara lista.")
        
        # --- Estado inicial ---
        controlar_banda(True) # Arrancar la banda
        print("\n======================================")
        print("✅ Sistema listo. Esperando producto...")
        print("======================================")
        
        estado_anterior_sensor = not SENSOR_DETECTA_EN
        
        # --- Bucle de detección ---
        while True:
            estado_sensor = GPIO.input(PIN_SENSOR_CAMARA)
            
            # --- Detección de flanco (solo al momento de detectar) ---
            # Comparamos el estado actual con el anterior para evitar
            # que clasifique 100 veces el mismo objeto.
            if estado_sensor == SENSOR_DETECTA_EN and estado_anterior_sensor != SENSOR_DETECTA_EN:
                
                print("\n--- ¡Producto detectado! ---")
                
                # 1. Detener banda
                controlar_banda(False)
                time.sleep(0.2) # Esperar a que la vibración pare
                
                # 2. Capturar imagen
                print("Capturando imagen...")
                ret, frame = cam.read()
                
                if not ret:
                    print("❌ Error al capturar imagen. Reanudando banda.")
                else:
                    # Guardar imagen (opcional, para debug)
                    # cv2.imwrite(f"captura_{int(time.time())}.jpg", frame)
                    
                    # 3. Clasificar
                    print("Procesando imagen...")
                    clase, conf = clasificar(frame)
                    print(f"➡️ Resultado: {clase} (Confianza: {conf*100:.1f}%)")
                    
                    # 4. Accionar Servos (Clasificar)
                    # Solo accionar si estamos seguros
                    if conf > 0.75: # Umbral de confianza del 75%
                        accionar_clasificador(clase)
                    else:
                        print(f"Confianza baja ({conf*100:.1f}%), dejando pasar.")
                        time.sleep(1.0) # Espera igual
                
                # 5. Reanudar banda
                print("Proceso terminado. Reanudando banda...")
                controlar_banda(True)
                
                # Esperar a que el producto salga del sensor
                print("Esperando que el producto libere el sensor...")
                while GPIO.input(PIN_SENSOR_CAMARA) == SENSOR_DETECTA_EN:
                    time.sleep(0.1)
                print("Sensor libre.")
            
            # Actualizar estado anterior
            estado_anterior_sensor = estado_sensor
            
            # Pequeña pausa para no saturar el CPU
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nCerrando sistema por el usuario...")
    
    finally:
        # --- Limpieza ---
        print("Apagando hardware.")
        controlar_banda(False)
        if cam:
            cam.release()
        if servo_cereza_pwm:
            servo_cereza_pwm.stop()
        if servo_fresa_pwm:
            servo_fresa_pwm.stop()
        GPIO.cleanup()
        print("Sistema apagado.")

if __name__ == "__main__":
    main()