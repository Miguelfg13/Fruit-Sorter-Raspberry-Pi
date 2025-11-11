🍓 Clasificador de Frutas en Cinta Transportadora (Raspberry Pi + TFLite)
Este proyecto es un sistema completo de visión artificial e IoT industrial para clasificar frutas (cerezas y fresas) en una banda transportadora. Utiliza un modelo de Deep Learning (MobileNetV2) entrenado con TensorFlow/Keras y desplegado en una Raspberry Pi para control de hardware en tiempo real.

El sistema identifica si la fruta está en buen o mal estado y acciona servomotores para desviarla a la línea de calidad correspondiente.

🚀 Características
Clasificación de 4 Clases: cereza_buena, cereza_mala, fresa_buena, fresa_mala.

Modelo de Alta Precisión: Logra +95% de precisión gracias a una arquitectura de transfer learning afinada.

Inferencia Eficiente: Utiliza un modelo TensorFlow Lite (.tflite) optimizado para una rápida ejecución en la CPU de la Raspberry Pi.

Control de Hardware (GPIO):

Lee un sensor de proximidad para detener la banda cuando llega una fruta.

Controla un relé para detener/arrancar el motor de la banda transportadora.

Controla dos servomotores para clasificar físicamente las frutas buenas.

🛠️ Hardware Requerido
Raspberry Pi (3B+, 4, o superior)

Webcam (USB o Módulo de Cámara Pi)

Sensor de proximidad (Ej. IR)

1x Relé de 1 canal (para el motor de la banda)

2x Servomotores (Ej. SG90 o MG90S)

Cables, protoboard y resistencias.

(Opcional pero recomendado) Un filtro polarizador para la lente de la cámara.

## 📂 Estructura del Repositorio

* `clasificador_banda.py` - Script principal de control e inferencia (Raspberry Pi)
* `train.py` - Script de entrenamiento (PC)
* `test_model.py` - Script de prueba del modelo .tflite (PC)
* `fruit_model.tflite` - EL MODELO FINAL (copiar a la Pi)
* `requirements_pc.txt` - Dependencias de Python para la PC (Entrenamiento)
* `requirements_pi.txt` - Dependencias de Python para la Raspberry Pi (Control)
* `README.md` - Este archivo
* `dataset/` (Ignorado por Git)
    * `train/`
        * `cereza_buena/`
        * `cereza_mala/`
        * `...`
    * `validation/`
        * `cereza_buena/`
        * `...`

⚙️ Software e Instalación
Este proyecto tiene dos entornos separados: la PC de Entrenamiento y la Raspberry Pi de Despliegue.

1. En tu PC (Para Entrenamiento)
Se usa para entrenar el modelo. Necesita la versión de escritorio completa de TensorFlow y OpenCV.

Bash

# Se recomienda usar un entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instala las dependencias
pip install -r requirements_pc.txt
Contenido de requirements_pc.txt: (¡Las versiones son importantes para evitar conflictos!)

Plaintext

tensorflow==2.19.0
numpy==2.1.3
opencv-python==4.12.0.88
matplotlib
scikit-learn
seaborn
2. En tu Raspberry Pi (Para Despliegue)
Se usa para correr la banda. Utiliza tflite-runtime (más ligero) y opencv-headless (sin interfaz gráfica).

Bash

# Instala las dependencias en tu Pi
pip3 install -r requirements_pi.txt
Contenido de requirements_pi.txt:

Plaintext

tflite-runtime
opencv-python-headless
numpy
RPi.GPIO
🏁 Flujo de Trabajo
Paso 1: Entrenamiento (en PC)
Prepara tu Dataset: Asegúrate de que tu carpeta dataset/ tenga la estructura de train y validation mostrada arriba.

Ejecuta el Entrenamiento:

Bash

python train.py
Resultado: El script entrenará el modelo, guardará gráficos y, lo más importante, creará fruit_model.tflite en la carpeta raíz.

Paso 2: Prueba (en PC)
Verifica que el modelo .tflite funciona correctamente antes de moverlo a la Pi.

Bash

# Probar con una imagen aleatoria del dataset
python test_model.py

# Probar con una imagen específica
python test_model.py "dataset/validation/fresa_buena/img_001.jpg"
El script mostrará la predicción en la terminal y abrirá una ventana de OpenCV con la imagen.

Paso 3: Despliegue (en Raspberry Pi)
Copia los archivos a tu Raspberry Pi (usando scp, una USB, etc.):

fruit_model.tflite

clasificador_banda.py

Configura los Pines: Abre clasificador_banda.py y ajusta los números de pin GPIO en la sección CONFIGURACIÓN DE HARDWARE para que coincidan con tus conexiones.

¡Ejecuta el Sistema!

Bash

python3 clasificador_banda.py
El script se inicializará, encenderá la banda y esperará a que el sensor detecte la primera fruta.

🚨 Consideraciones Clave y Lecciones Aprendidas
Esta sección documenta los problemas críticos encontrados y sus soluciones, sirviendo como registro de optimización.

1. EL ERROR CRÍTICO: Doble Normalización (Precisión del 31%)
Problema: El modelo inicial tenía una precisión de solo 31%.

Causa: Estábamos normalizando los datos dos veces.

ImageDataGenerator(rescale=1./255) convertía los píxeles de [0, 255] a [0, 1].

La capa keras.applications.mobilenet_v2.preprocess_input (que está dentro del modelo) esperaba [0, 255] para convertirlos a [-1, 1].

Solución: Eliminar rescale=1./255 del ImageDataGenerator. El modelo TFLite ahora incluye la capa de preprocesamiento y solo espera la imagen en bruto (formato float32 de 0 a 255).

2. El "Infierno de Dependencias" en PC
Problema: tensorflow y opencv-python tienen requisitos de numpy conflictivos.

Contexto: tensorflow 2.19.0 requiere numpy < 2.2.0, pero opencv-python 4.12.0 requiere numpy >= 2.0.

Solución: Encontrar una versión "puente". Se determinó que numpy==2.1.3 es compatible con ambos paquetes. Las versiones exactas están fijadas en requirements_pc.txt.

3. Error cv2.imshow (Headless vs. Desktop)
Problema: El script test_model.py fallaba con error: The function is not implemented... in function 'cvShowImage'.

Causa: Instalación de opencv-python-headless. Esta versión no incluye soporte para interfaces gráficas (GUI) y no puede abrir ventanas.

Solución: La PC de desarrollo DEBE usar opencv-python (la versión de escritorio completa). La Raspberry Pi DEBE usar opencv-python-headless (más ligera, no necesita GUI).

4. tflite-runtime vs. tensorflow.lite
Problema: El script test_model.py fallaba en la PC con ModuleNotFoundError: No module named 'tflite_runtime'.

Causa: tflite-runtime es un paquete separado solo para inferencia (usado en la Pi). El paquete completo de tensorflow (usado en la PC) contiene esta funcionalidad en tensorflow.lite.

Solución: Los scripts (test_model.py y clasificador_banda.py) usan un bloque try/except para importar el módulo correcto según el entorno, haciéndolos portables.

5. Consideraciones Físicas (Hardware)
Problema: Las imágenes de la banda transportadora muestran muchos reflejos de luz (brillo especular) sobre la fruta y el metal.

Riesgo: Un reflejo blanco puede ser confundido por la red con un hongo o una "mancha mala", o puede ocultar defectos reales.

Solución (Recomendada):

Filtro Polarizador: Colocar un filtro polarizador en la lente de la cámara es la mejor solución para eliminar casi todos los reflejos.

Iluminación Difusa: Usar luces más suaves o rebotadas en lugar de un foco directo.

6. Arquitectura del Modelo
Problema: Un modelo con muchas capas densas (Dense(256), Dense(128)) después del modelo base (MobileNetV2) puede ser propenso a sobreajuste (overfitting), especialmente con un dataset industrial pequeño.

Solución: La arquitectura del modelo fue simplificada a:

base_model (MobileNetV2)

GlobalAveragePooling2D()

Dropout(0.5) (Para regularización fuerte)

Dense(4, activation='softmax') (Capa de salida directa)

Resultado: Un modelo más ligero, más rápido y que generaliza mejor.
