
# 🍓 Clasificador de Frutas en Cinta Transportadora (Raspberry Pi + TFLite)

Este proyecto es un sistema completo de **visión artificial e IoT industrial** para clasificar frutas (cerezas y fresas) en una banda transportadora.  
Utiliza un modelo de *Deep Learning* (**MobileNetV2**) entrenado con **TensorFlow/Keras** y desplegado en una **Raspberry Pi** para el control de hardware en tiempo real.

El sistema **identifica si la fruta está en buen o mal estado** y acciona **servomotores** para desviarla a la línea de calidad correspondiente.

---

## 🚀 Características

- **Clasificación de 4 clases:**  
  `cereza_buena`, `cereza_mala`, `fresa_buena`, `fresa_mala`.
- **Modelo de alta precisión:**  
  Logra más del **95% de exactitud** gracias al uso de *transfer learning*.
- **Inferencia eficiente:**  
  Usa un modelo **TensorFlow Lite (.tflite)** optimizado para ejecución en CPU.
- **Control de hardware (GPIO):**
  - Lee un **sensor de proximidad** para detener la banda al detectar una fruta.  
  - Controla un **relé** para detener/arrancar el motor de la banda.  
  - Controla **dos servomotores** para clasificar las frutas buenas.

---

## 🛠️ Hardware Requerido

- Raspberry Pi (3B+, 4 o superior)
- Webcam (USB o módulo de cámara Pi)
- Sensor de proximidad (Ej. IR)
- 1x Relé de 1 canal (para el motor)
- 2x Servomotores (Ej. SG90 o MG90S)
- Cables, protoboard y resistencias
- *(Opcional pero recomendado)* Filtro polarizador para la cámara

---

## 📂 Estructura del Repositorio

```

├── clasificador_banda.py        # Script principal de control e inferencia (Raspberry Pi)
├── train.py                     # Entrenamiento del modelo (PC)
├── test_model.py                # Prueba del modelo .tflite (PC)
├── fruit_model.tflite           # Modelo final (copiar a la Pi)
├── requirements_pc.txt          # Dependencias para PC (entrenamiento)
├── requirements_pi.txt          # Dependencias para Raspberry Pi (despliegue)
├── README.md                    # Este archivo
└── dataset/                     # Dataset (ignorado por Git)
├── train/
│   ├── cereza_buena/
│   ├── cereza_mala/
│   └── ...
└── validation/
├── cereza_buena/
└── ...

````

---

## ⚙️ Software e Instalación

Este proyecto se divide en dos entornos:

1. **PC (Entrenamiento del modelo)**
2. **Raspberry Pi (Despliegue en tiempo real)**

---

### 🧠 1. En tu PC — *Entrenamiento del modelo*

Se usa para entrenar la red neuronal con TensorFlow y OpenCV.

```bash
# Recomendado: crear un entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements_pc.txt
````

**Contenido de `requirements_pc.txt`:**

```plaintext
tensorflow==2.19.0
numpy==2.1.3
opencv-python==4.12.0.88
matplotlib
scikit-learn
seaborn
```

---

### 🤖 2. En tu Raspberry Pi — *Despliegue del modelo*

Usa una versión ligera del runtime de TensorFlow (tflite-runtime) y OpenCV sin interfaz gráfica.

```bash
# Instalar dependencias
pip3 install -r requirements_pi.txt
```

**Contenido de `requirements_pi.txt`:**

```plaintext
tflite-runtime
opencv-python-headless
numpy
RPi.GPIO
```

---

## 🏁 Flujo de Trabajo

### 🔹 Paso 1: Entrenamiento (PC)

1. Prepara tu carpeta `dataset/` con la estructura `train/` y `validation/`.
2. Ejecuta el entrenamiento:

   ```bash
   python train.py
   ```

   **Resultado:**
   Se guardará el modelo `fruit_model.tflite` en la carpeta raíz junto con los gráficos de rendimiento.

---

### 🔹 Paso 2: Prueba del Modelo (PC)

Verifica el modelo antes de transferirlo a la Pi:

```bash
# Probar con una imagen aleatoria del dataset
python test_model.py

# O con una imagen específica
python test_model.py "dataset/validation/fresa_buena/img_001.jpg"
```

El script mostrará la predicción y abrirá una ventana de OpenCV con la imagen.

---

### 🔹 Paso 3: Despliegue (Raspberry Pi)

1. Copia los siguientes archivos a tu Raspberry Pi:

   * `fruit_model.tflite`
   * `clasificador_banda.py`

2. Configura los pines GPIO en `clasificador_banda.py` según tus conexiones físicas.

3. Ejecuta el sistema:

   ```bash
   python3 clasificador_banda.py
   ```

El sistema se inicializará, encenderá la banda y esperará a que el sensor detecte la primera fruta.

---

## 🚨 Consideraciones Clave y Lecciones Aprendidas

### 1️⃣ Doble Normalización (Precisión del 31%)

* **Problema:** El modelo inicial tenía una precisión baja (~31%).
* **Causa:** Se aplicó normalización doble (`rescale=1./255` + `preprocess_input`).
* **Solución:** Eliminar `rescale=1./255` del `ImageDataGenerator`.
  El modelo ahora espera imágenes en formato float32 [0,255].

---

### 2️⃣ Conflictos de Dependencias (PC)

* **Problema:** `tensorflow` y `opencv-python` exigían versiones distintas de `numpy`.
* **Solución:** Usar `numpy==2.1.3`, compatible con ambas librerías.

---

### 3️⃣ Error `cv2.imshow` (Entornos Headless)

* **Causa:** La versión `opencv-python-headless` no soporta GUI.
* **Solución:**

  * En PC: usar `opencv-python`
  * En Raspberry Pi: usar `opencv-python-headless`

---

### 4️⃣ `tflite-runtime` vs `tensorflow.lite`

* **Problema:** `ModuleNotFoundError: No module named 'tflite_runtime'` en PC.
* **Solución:**
  Los scripts usan un bloque `try/except` para importar automáticamente la versión correcta.

---

### 5️⃣ Consideraciones Físicas (Hardware)

* **Problema:** Reflejos de luz en la fruta o el metal.
* **Riesgo:** Los reflejos pueden confundir a la red o ocultar defectos.
* **Soluciones:**

  * Usar **filtro polarizador** en la cámara.
  * Emplear **iluminación difusa** o rebotada.

---

### 6️⃣ Arquitectura del Modelo

**Problema:** Modelos con muchas capas densas provocaban *overfitting*.
**Solución:** Simplificar la arquitectura:

```python
base_model = MobileNetV2(...)
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.5)(x)
outputs = Dense(4, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=outputs)
```

**Resultado:** Modelo más ligero, rápido y con mejor generalización.

---

## 📸 Créditos y Licencia

Proyecto desarrollado por **Miguel Flores**  

[![Licencia MIT](https://img.shields.io/badge/Licencia-MIT-green.svg)](LICENSE)
![Hecho con ❤️](https://img.shields.io/badge/Hecho%20con-%E2%9D%A4-red)
![TensorFlow](https://img.shields.io/badge/TensorFlow-orange)
![Raspberry%20Pi](https://img.shields.io/badge/Raspberry%20Pi-red)


