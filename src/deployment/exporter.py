"""
Funciones para exportar modelos a diferentes formatos para despliegue.
"""

import os
import tensorflow as tf
import time
 
def export_saved_model(model, output_dir):
    """
    Exporta el modelo al formato SavedModel de TensorFlow.
    
    Args:
        model (tf.keras.Model): Modelo a exportar
        output_dir (str): Directorio donde guardar el modelo
    """
    print("Exportando modelo en formato SavedModel...")
    start_time = time.time()
    
    # Crear directorio si no existe
    saved_model_dir = os.path.join(output_dir, "saved_model")
    os.makedirs(saved_model_dir, exist_ok=True)
    
    # Guardar modelo
    model.save(saved_model_dir)
    
    print(f"Modelo exportado en {time.time() - start_time:.2f} segundos a: {saved_model_dir}")
    
    # También guardar en formato H5
    h5_path = os.path.join(output_dir, "model.h5")
    model.save(h5_path)
    print(f"Modelo guardado en formato H5: {h5_path}")

def export_tflite_model(model, output_dir, quantize=True):
    """
    Exporta el modelo al formato TFLite.
    
    Args:
        model (tf.keras.Model): Modelo a exportar
        output_dir (str): Directorio donde guardar el modelo
        quantize (bool): Si se debe cuantizar el modelo
    """
    print("Exportando modelo en formato TFLite...")
    start_time = time.time()
    
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Crear convertidor
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Modelo estándar
    tflite_model = converter.convert()
    tflite_model_path = os.path.join(output_dir, "model.tflite")
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Modelo TFLite guardado en: {tflite_model_path}")
    
    # Modelo cuantizado (si se solicita)
    if quantize:
        print("Creando modelo TFLite cuantizado...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_quantized_model = converter.convert()
        tflite_quantized_path = os.path.join(output_dir, "model_quantized.tflite")
        with open(tflite_quantized_path, 'wb') as f:
            f.write(tflite_quantized_model)
        
        print(f"Modelo TFLite cuantizado guardado en: {tflite_quantized_path}")
    
    print(f"Exportación TFLite completada en {time.time() - start_time:.2f} segundos.")