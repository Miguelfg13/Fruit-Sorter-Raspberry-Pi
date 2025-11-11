
"""
Script para capturar imágenes y crear dataset de entrenamiento
PARA PC CON WEBCAM - Compatible con Windows/Linux/Mac
"""

import cv2
import os
from datetime import datetime
import time
import numpy as np

# ==================== CONFIGURACIÓN ====================
DATASET_DIR = 'dataset'
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
VAL_DIR = os.path.join(DATASET_DIR, 'validation')

CLASSES = ['cereza_buena', 'cereza_mala', 'fresa_buena', 'fresa_mala']

# Configuración de cámara
CAMERA_INDEX = 1  # 0 = cámara por defecto, cambiar si tienes múltiples cámaras
RESOLUTION_WIDTH = 1280
RESOLUTION_HEIGHT = 720


# ==================== SETUP ====================
def crear_estructura_carpetas():
    """Crea la estructura de carpetas del dataset"""
    for split in [TRAIN_DIR, VAL_DIR]:
        for clase in CLASSES:
            path = os.path.join(split, clase)
            os.makedirs(path, exist_ok=True)
    
    print("✅ Estructura de carpetas creada:")
    print(f"   {DATASET_DIR}/")
    print(f"   ├── train/")
    for clase in CLASSES:
        print(f"   │   ├── {clase}/")
    print(f"   └── validation/")
    for clase in CLASSES:
        print(f"       ├── {clase}/")


# ==================== CAPTURA CON WEBCAM ====================
class CapturaDatasetPC:
    def __init__(self, camera_index=1):
        """Inicializa la webcam y contadores"""
        print("\n🎥 Inicializando webcam...")
        
        # Intentar abrir la cámara
        self.cap = cv2.VideoCapture(1)
        
        if not self.cap.isOpened():
            raise Exception(f"❌ No se pudo abrir la cámara {camera_index}")
        
        # Configurar resolución
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION_HEIGHT)
        
        # Verificar resolución real
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"✅ Webcam iniciada correctamente")
        print(f"📐 Resolución: {width}x{height}")
        
        # Inicializar contadores
        self.contadores = {clase: {'train': 0, 'val': 0} for clase in CLASSES}
        self._cargar_contadores_existentes()
        
        # Variables para modo preview
        self.preview_activo = False
        self.clase_actual = None
        self.split_actual = 'train'
        
    def _cargar_contadores_existentes(self):
        """Cuenta imágenes existentes en cada carpeta"""
        for clase in CLASSES:
            # Contar train
            train_path = os.path.join(TRAIN_DIR, clase)
            if os.path.exists(train_path):
                self.contadores[clase]['train'] = len([f for f in os.listdir(train_path) 
                                                       if f.endswith('.jpg')])
            
            # Contar validation
            val_path = os.path.join(VAL_DIR, clase)
            if os.path.exists(val_path):
                self.contadores[clase]['val'] = len([f for f in os.listdir(val_path) 
                                                     if f.endswith('.jpg')])
    
    def capturar_frame(self):
        """Captura un frame de la webcam"""
        ret, frame = self.cap.read()
        if not ret:
            print("❌ Error al capturar frame")
            return None
        return frame
    
    def guardar_imagen(self, frame, clase, split='train'):
        """
        Guarda una imagen capturada
        
        Args:
            frame: Frame de OpenCV
            clase: Clase de la fruta
            split: 'train' o 'val'
        
        Returns:
            bool: True si se guardó correctamente
        """
        if clase not in CLASSES:
            print(f"❌ Clase inválida: {clase}")
            return False
        
        # Generar nombre de archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        self.contadores[clase][split] += 1
        count = self.contadores[clase][split]
        filename = f"{clase}_{timestamp}_{count:04d}.jpg"
        
        # Determinar ruta
        if split == 'train':
            path = os.path.join(TRAIN_DIR, clase, filename)
        else:
            path = os.path.join(VAL_DIR, clase, filename)
        
        # Guardar imagen
        cv2.imwrite(path, frame)
        print(f"📸 Capturada: {filename} → {split}/{clase}/ (Total: {count})")
        return True
    
    def modo_visual_interactivo(self):
        """
        Modo interactivo con preview visual de la cámara
        Usa la ventana de OpenCV para capturar
        """
        print("\n" + "="*70)
        print("📸 MODO VISUAL INTERACTIVO")
        print("="*70)
        print("\n🎯 Controles en ventana de video:")
        print("  ESPACIO    → Capturar imagen")
        print("  1,2,3,4    → Seleccionar clase (cereza_buena, cereza_mala, etc.)")
        print("  T          → Cambiar a Train")
        print("  V          → Cambiar a Validation")
        print("  S          → Ver estadísticas")
        print("  Q / ESC    → Salir")
        print("\n📊 Clases:")
        for i, clase in enumerate(CLASSES, 1):
            print(f"  {i}. {clase}")
        print("="*70)
        
        self.clase_actual = CLASSES[0]
        self.split_actual = 'train'
        
        print(f"\n🎬 Iniciando preview... Presiona Q o ESC para salir")
        
        while True:
            frame = self.capturar_frame()
            if frame is None:
                break
            
            # Crear copia para overlay
            display = frame.copy()
            h, w = display.shape[:2]
            
            # Overlay con información
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
            
            # Textos informativos
            cv2.putText(display, f"Clase: {self.clase_actual}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display, f"Split: {self.split_actual.upper()}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            count = self.contadores[self.clase_actual][self.split_actual]
            cv2.putText(display, f"Total: {count}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Guía de teclas
            cv2.putText(display, "ESPACIO=Capturar | 1-4=Clase | T/V=Split | Q=Salir", 
                       (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Mostrar frame
            cv2.imshow('Dataset Capture - Frutas', display)
            
            # Procesar teclas
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q o ESC
                print("\n👋 Saliendo del modo visual...")
                break
            
            elif key == ord(' '):  # ESPACIO - capturar
                self.guardar_imagen(frame, self.clase_actual, self.split_actual)
                # Flash visual
                flash = np.ones_like(display) * 255
                cv2.imshow('Dataset Capture - Frutas', flash)
                cv2.waitKey(100)
            
            elif key == ord('1'):
                self.clase_actual = CLASSES[0]
                print(f"📌 Clase cambiada a: {self.clase_actual}")
            
            elif key == ord('2'):
                self.clase_actual = CLASSES[1]
                print(f"📌 Clase cambiada a: {self.clase_actual}")
            
            elif key == ord('3'):
                self.clase_actual = CLASSES[2]
                print(f"📌 Clase cambiada a: {self.clase_actual}")
            
            elif key == ord('4'):
                self.clase_actual = CLASSES[3]
                print(f"📌 Clase cambiada a: {self.clase_actual}")
            
            elif key == ord('t'):
                self.split_actual = 'train'
                print(f"📂 Split cambiado a: TRAIN")
            
            elif key == ord('v'):
                self.split_actual = 'val'
                print(f"📂 Split cambiado a: VALIDATION")
            
            elif key == ord('s'):
                self.mostrar_estadisticas()
        
        cv2.destroyAllWindows()
        self.cerrar()
    
    def modo_automatico(self, clase, cantidad=100, split='train', intervalo=2.0, 
                       mostrar_preview=True):
        """
        Captura automática con preview opcional
        
        Args:
            clase: Clase de la fruta
            cantidad: Número de imágenes
            split: 'train' o 'val'
            intervalo: Segundos entre capturas
            mostrar_preview: Mostrar ventana con preview
        """
        print("\n" + "="*70)
        print(f"🤖 MODO AUTOMÁTICO")
        print("="*70)
        print(f"📌 Clase: {clase}")
        print(f"📊 Cantidad: {cantidad} imágenes")
        print(f"📂 Destino: {split}")
        print(f"⏱️  Intervalo: {intervalo} segundos")
        print(f"👁️  Preview: {'Activado' if mostrar_preview else 'Desactivado'}")
        print("\n⚠️  Presiona 'Q' o ESC para detener\n")
        
        capturadas = 0
        
        try:
            for i in range(cantidad):
                frame = self.capturar_frame()
                if frame is None:
                    print("❌ Error al capturar frame")
                    break
                
                # Mostrar preview si está activado
                if mostrar_preview:
                    display = frame.copy()
                    h, w = display.shape[:2]
                    
                    # Overlay
                    text = f"Capturando {i+1}/{cantidad} - {clase} ({split})"
                    cv2.putText(display, text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display, "Presiona Q para detener", (10, h-20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                    
                    cv2.imshow('Captura Automatica', display)
                    
                    # Verificar si se presionó Q o ESC
                    if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                        print("\n⚠️  Captura detenida por el usuario")
                        break
                
                # Guardar imagen
                if self.guardar_imagen(frame, clase, split):
                    capturadas += 1
                
                # Mostrar progreso
                if (i + 1) % 10 == 0:
                    progreso = ((i + 1) / cantidad) * 100
                    print(f"📊 Progreso: {i + 1}/{cantidad} ({progreso:.1f}%)")
                
                # Esperar intervalo
                if i < cantidad - 1:
                    time.sleep(intervalo)
        
        except KeyboardInterrupt:
            print(f"\n⚠️  Captura detenida por el usuario")
        finally:
            if mostrar_preview:
                cv2.destroyAllWindows()
        
        print(f"\n✅ Completado: {capturadas}/{cantidad} imágenes capturadas")
        print(f"📁 Guardadas en: {split}/{clase}/")
    
    def mostrar_estadisticas(self):
        """Muestra estadísticas actuales de captura"""
        print("\n" + "="*70)
        print("📊 ESTADÍSTICAS DE CAPTURA")
        print("="*70)
        print(f"{'Clase':<20} {'Train':>12} {'Validation':>15} {'Total':>12}")
        print("-"*70)
        
        total_train = 0
        total_val = 0
        
        for clase in CLASSES:
            train_count = self.contadores[clase]['train']
            val_count = self.contadores[clase]['val']
            total = train_count + val_count
            
            total_train += train_count
            total_val += val_count
            
            print(f"{clase:<20} {train_count:>12} {val_count:>15} {total:>12}")
        
        print("-"*70)
        print(f"{'TOTAL':<20} {total_train:>12} {total_val:>15} {total_train + total_val:>12}")
        print("="*70)
        
        # Recomendaciones
        print("\n💡 Recomendaciones:")
        for clase in CLASSES:
            train_count = self.contadores[clase]['train']
            val_count = self.contadores[clase]['val']
            
            if train_count < 100:
                print(f"   ⚠️  {clase}: Necesitas más imágenes de entrenamiento (mínimo 100, ideal 150+)")
            if val_count < 30:
                print(f"   ⚠️  {clase}: Necesitas más imágenes de validación (mínimo 30, ideal 40+)")
    
    def cerrar(self):
        """Cierra la cámara y muestra estadísticas finales"""
        print("\n🔄 Cerrando webcam...")
        self.cap.release()
        cv2.destroyAllWindows()
        self.mostrar_estadisticas()
        print("\n✅ Sistema cerrado correctamente\n")


# ==================== MAIN ====================
def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Herramienta de captura de dataset para PC con webcam',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  Modo visual interactivo (RECOMENDADO):
    python capture_dataset_pc.py

  Modo automático con preview:
    python capture_dataset_pc.py --modo automatico --clase cereza_buena --cantidad 150

  Modo automático sin preview (más rápido):
    python capture_dataset_pc.py --modo automatico --clase fresa_mala --cantidad 100 --no-preview

  Especificar cámara (si tienes múltiples):
    python capture_dataset_pc.py --camera 1

  Captura rápida (intervalo corto):
    python capture_dataset_pc.py --modo automatico --clase cereza_buena --cantidad 50 --intervalo 0.5
        """
    )
    
    parser.add_argument(
        '--modo',
        choices=['visual', 'automatico'],
        default='visual',
        help='Modo de captura (default: visual)'
    )
    
    parser.add_argument(
        '--clase',
        choices=CLASSES,
        help='Clase para modo automático'
    )
    
    parser.add_argument(
        '--cantidad',
        type=int,
        default=100,
        help='Cantidad de imágenes en modo automático (default: 100)'
    )
    
    parser.add_argument(
        '--split',
        choices=['train', 'val'],
        default='train',
        help='Destino: train o val (default: train)'
    )
    
    parser.add_argument(
        '--intervalo',
        type=float,
        default=2.0,
        help='Intervalo en segundos entre capturas (default: 2.0)'
    )
    
    parser.add_argument(
        '--camera',
        type=int,
        default=1,
        help='Índice de cámara (default: 0)'
    )
    
    parser.add_argument(
        '--no-preview',
        action='store_true',
        help='Desactivar preview en modo automático (más rápido)'
    )
    
    args = parser.parse_args()
    
    # Banner
    print("\n" + "="*70)
    print("🍓🍒 CAPTURA DE DATASET - CLASIFICACIÓN DE FRUTAS (PC)")
    print("="*70)
    
    # Crear estructura de carpetas
    crear_estructura_carpetas()
    
    # Validar modo automático
    if args.modo == 'automatico' and not args.clase:
        print("\n❌ Error: Debes especificar --clase en modo automático")
        print("   Ejemplo: --clase cereza_buena")
        print(f"   Clases válidas: {', '.join(CLASSES)}")
        return
    
    # Iniciar captura
    try:
        captura = CapturaDatasetPC(camera_index=args.camera)
        
        if args.modo == 'visual':
            captura.modo_visual_interactivo()
        else:
            captura.modo_automatico(
                clase=args.clase,
                cantidad=args.cantidad,
                split=args.split,
                intervalo=args.intervalo,
                mostrar_preview=not args.no_preview
            )
            captura.cerrar()
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Programa interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()