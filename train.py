#!/usr/bin/env python3
"""
Script para entrenar modelo de clasificación de frutas - VERSIÓN CORREGIDA
OPTIMIZADO PARA PC - Entrena con TensorFlow y convierte a TFLite para Raspberry Pi
Compatible con Windows/Linux/Mac
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import json
from datetime import datetime

# ==================== CONFIGURACIÓN ====================
IMG_SIZE = 224  # Tamaño de entrada de MobileNetV2
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001 # Aumentado ligeramente para la fase 1

# Estructura de carpetas esperada:
DATASET_DIR = 'dataset'
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
VAL_DIR = os.path.join(DATASET_DIR, 'validation')

CLASSES = ['cereza_buena', 'cereza_mala', 'fresa_buena', 'fresa_mala']

# Carpeta de salida
OUTPUT_DIR = 'modelos_entrenados'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==================== UTILIDADES ====================
def verificar_dataset():
    """Verifica que el dataset tenga suficientes imágenes"""
    print("\n🔍 Verificando dataset...")
    
    if not os.path.exists(TRAIN_DIR):
        print(f"❌ Error: No existe la carpeta {TRAIN_DIR}")
        return False
    
    if not os.path.exists(VAL_DIR):
        print(f"❌ Error: No existe la carpeta {VAL_DIR}")
        return False
    
    print("\n📊 Resumen del dataset:")
    print("-" * 60)
    print(f"{'Clase':<20} {'Train':>12} {'Validation':>15}")
    print("-" * 60)
    
    dataset_valido = True
    total_train = 0
    total_val = 0
    
    for clase in CLASSES:
        train_path = os.path.join(TRAIN_DIR, clase)
        val_path = os.path.join(VAL_DIR, clase)
        
        if not os.path.exists(train_path):
            print(f"❌ Falta carpeta: {train_path}")
            dataset_valido = False
            continue
        
        if not os.path.exists(val_path):
            print(f"❌ Falta carpeta: {val_path}")
            dataset_valido = False
            continue
        
        train_count = len([f for f in os.listdir(train_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
        val_count = len([f for f in os.listdir(val_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        total_train += train_count
        total_val += val_count
        
        print(f"{clase:<20} {train_count:>12} {val_count:>15}")
        
        if train_count < 50:
            print(f"   ⚠️  Advertencia: Pocas imágenes de entrenamiento (mínimo recomendado: 100)")
        if val_count < 20:
            print(f"   ⚠️  Advertencia: Pocas imágenes de validación (mínimo recomendado: 30)")
    
    print("-" * 60)
    print(f"{'TOTAL':<20} {total_train:>12} {total_val:>15}")
    print("-" * 60)
    
    if total_train == 0 or total_val == 0:
        print("\n❌ Error: Dataset vacío")
        return False
    
    if total_train < 200:
        print("\n⚠️  Advertencia: Dataset pequeño. Recomendado: 100+ imágenes por clase")
    
    return dataset_valido


# ==================== PREPARACIÓN DE DATOS ====================
def crear_dataset():
    """Crea los datasets de entrenamiento y validación con data augmentation"""
    
    print("\n📂 Preparando datasets...")
    
    # CORRECCIÓN CRÍTICA: Eliminado rescale=1./255 porque MobileNetV2 incluye su propio preprocesamiento
    # Augmentación ajustada para ser menos agresiva con cintas transportadoras
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,      # Reducido de 30 a 15
        width_shift_range=0.1,  # Reducido de 0.3 a 0.1
        height_shift_range=0.1, # Reducido de 0.3 a 0.1
        shear_range=0.1,        # Reducido de 0.2 a 0.1
        zoom_range=0.2,         # Reducido de 0.3 a 0.2
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2] # Rango de brillo ligeramente reducido
    )
    
    # CORRECCIÓN CRÍTICA: Eliminado rescale=1./255
    val_datagen = keras.preprocessing.image.ImageDataGenerator()
    
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASSES,
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASSES,
        shuffle=False
    )
    
    print(f"✅ Dataset de entrenamiento: {train_generator.samples} imágenes")
    print(f"✅ Dataset de validación: {val_generator.samples} imágenes")
    
    return train_generator, val_generator


# ==================== CONSTRUCCIÓN DEL MODELO ====================
def crear_modelo():
    """Crea modelo basado en MobileNetV2 con transfer learning"""
    
    print("\n🏗️  Construyendo modelo...")
    
    # Cargar MobileNetV2 preentrenado (sin la capa superior)
    # MobileNetV2 espera inputs [-1, 1]. Usaremos su capa de preprocesamiento.
    base_model = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Congelar las capas del modelo base inicialmente
    base_model.trainable = False
    
    # Construir modelo completo
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Preprocesamiento específico de MobileNetV2 (convierte [0,255] a [-1,1])
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    
    # Base model
    x = base_model(x, training=False)
    
    # Capas de clasificación SIMPLIFICADAS para evitar overfitting en datasets pequeños
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)  # Dropout más fuerte para regularización
    
    # Capa de salida directa (eliminamos las capas intermedias complejas)
    outputs = layers.Dense(len(CLASSES), activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    return model, base_model


# ==================== ENTRENAMIENTO ====================
def entrenar_modelo(model, base_model, train_gen, val_gen):
    """Entrena el modelo en dos fases"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(OUTPUT_DIR, f'training_{timestamp}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("📚 FASE 1: Entrenamiento de capas superiores (Transfer Learning)")
    print("="*70)
    
    # Compilar modelo
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks_phase1 = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(checkpoint_dir, 'best_model_phase1.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.CSVLogger(
            os.path.join(checkpoint_dir, 'training_log_phase1.csv')
        )
    ]
    
    # Entrenar fase 1
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20, # Menos epochs para la primera fase
        callbacks=callbacks_phase1,
        verbose=1
    )
    
    print("\n" + "="*70)
    print("🔥 FASE 2: Fine-tuning (Ajuste fino)")
    print("="*70)
    
    # Descongelar últimas capas del modelo base para fine-tuning
    base_model.trainable = True
    
    # Congelar las primeras capas, descongelar SOLO las últimas capas superiores
    # MobileNetV2 tiene 155 capas. Descongelamos las últimas 50.
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    print(f"📌 Capas totales base: {len(base_model.layers)}")
    print(f"📌 Capas congeladas: {fine_tune_at}")
    print(f"📌 Capas entrenables para fine-tuning: {len(base_model.layers) - fine_tune_at}")
    
    # Recompilar con tasa de aprendizaje MUY baja para no destruir lo aprendido
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5), # 0.00001
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks_phase2 = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(checkpoint_dir, 'best_model_phase2.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.CSVLogger(
            os.path.join(checkpoint_dir, 'training_log_phase2.csv')
        )
    ]
    
    # Entrenar fase 2
    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS, # Resto de epochs
        callbacks=callbacks_phase2,
        initial_epoch=len(history1.history['loss']),
        verbose=1
    )
    
    return history1, history2, checkpoint_dir


# ==================== CONVERSIÓN A TFLITE ====================
def convertir_a_tflite(model, output_dir):
    """Convierte el modelo a TensorFlow Lite optimizado para Raspberry Pi"""
    
    print("\n" + "="*70)
    print("📦 Convirtiendo modelo a TensorFlow Lite...")
    print("="*70)
    
    # Ruta de salida
    tflite_path = os.path.join(output_dir, 'fruit_model.tflite')
    
    # Convertir a TFLite con optimizaciones
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optimizaciones para Raspberry Pi
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # NOTA: Para máxima compatibilidad, a veces es mejor evitar SELECT_TF_OPS si es posible.
    # Probamos primero sin ops adicionales, si falla, descomentar las líneas de abajo.
    # converter.target_spec.supported_ops = [
    #     tf.lite.OpsSet.TFLITE_BUILTINS,
    #     tf.lite.OpsSet.SELECT_TF_OPS
    # ]
    
    print("⚙️  Aplicando optimización (cuantización dinámica)...")
    tflite_model = converter.convert()
    
    # Guardar modelo
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    file_size = os.path.getsize(tflite_path) / (1024 * 1024)
    print(f"✅ Modelo TFLite guardado en: {tflite_path}")
    print(f"📦 Tamaño del modelo: {file_size:.2f} MB")
    
    # También guardar en la raíz para fácil acceso
    root_tflite = 'fruit_model.tflite'
    with open(root_tflite, 'wb') as f:
        f.write(tflite_model)
    print(f"✅ Copia guardada en: {root_tflite}")
    
    return tflite_path


# ==================== VISUALIZACIÓN ====================
def graficar_resultados(history1, history2, output_dir):
    """Genera gráficas de entrenamiento"""
    
    print("\n📊 Generando gráficas...")
    
    # Combinar historias
    acc = history1.history['accuracy'] + history2.history['accuracy']
    val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    loss = history1.history['loss'] + history2.history['loss']
    val_loss = history1.history['val_loss'] + history2.history['val_loss']
    
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(12, 6))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.axvline(x=len(history1.history['accuracy'])-0.5, color='gray', linestyle='--', label='Inicio Fine-tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.grid(True)
    
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.axvline(x=len(history1.history['loss'])-0.5, color='gray', linestyle='--', label='Inicio Fine-tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Guardar
    plot_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    print(f"✅ Gráficas guardadas en: {plot_path}")


# ==================== EVALUACIÓN ====================
def evaluar_modelo(model, val_gen, output_dir):
    """Evalúa el modelo en el conjunto de validación"""
    
    print("\n" + "="*70)
    print("📈 EVALUACIÓN FINAL")
    print("="*70)
    
    results = model.evaluate(val_gen, verbose=1)
    
    print(f"\n📊 Métricas finales:")
    print(f"   Loss: {results[0]:.4f}")
    print(f"   Accuracy: {results[1]:.4f} ({results[1]*100:.2f}%)")
    
    # Predicciones
    print("\n🔮 Generando predicciones para matriz de confusión...")
    predictions = model.predict(val_gen, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = val_gen.classes
    
    # Matriz de confusión y reporte
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(true_classes, predicted_classes)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()
    
    # Reporte de clasificación
    report = classification_report(true_classes, predicted_classes, 
                                   target_names=CLASSES, digits=4, zero_division=0)
    print("\n📊 Reporte de clasificación:")
    print(report)
    
    # Guardar reporte
    report_path = os.path.join(output_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Accuracy: {results[1]:.4f}\n\n")
        f.write(report)
    
    return results


# ==================== MAIN ====================
def main():
    """Función principal"""
    
    print("\n" + "="*70)
    print("🍓🍒 ENTRENAMIENTO DE CLASIFICACIÓN DE FRUTAS v2 (CORREGIDO)")
    print("="*70)
    
    # Limitar uso de memoria GPU si existe para evitar errores OOM
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU configurada: {len(gpus)} dispositivo(s)")
        except RuntimeError as e:
            print(e)
    
    if not verificar_dataset():
        return
    
    train_gen, val_gen = crear_dataset()
    model, base_model = crear_modelo()
    
    print("\n📋 Modelo compilado. Iniciando entrenamiento...")
    history1, history2, checkpoint_dir = entrenar_modelo(model, base_model, train_gen, val_gen)
    
    eval_results = evaluar_modelo(model, val_gen, checkpoint_dir)
    graficar_resultados(history1, history2, checkpoint_dir)
    convertir_a_tflite(model, checkpoint_dir)
    
    print("\n✅ PROCESO COMPLETADO.")

if __name__ == "__main__":
    main()