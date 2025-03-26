"""
Funciones para evaluar el modelo de detección de objetos.
"""

import time
import os 
from src.data.visualization import visualize_predictions

def evaluate_model(model, test_dataset, class_names, readable_class_names, 
                  img_size=(416, 416), output_dir='predictions'):
    """
    Evalúa el modelo en el conjunto de prueba y muestra ejemplos de predicción.
    
    Args:
        model (tf.keras.Model): Modelo entrenado
        test_dataset (tf.data.Dataset): Conjunto de datos de prueba
        class_names (list): Lista de IDs de clases
        readable_class_names (list): Lista de nombres de clases legibles
        img_size (tuple): Tamaño de imagen (altura, ancho)
        output_dir (str): Directorio para guardar predicciones
    """
    print("\nEvaluando modelo en conjunto de prueba...")
    start_time = time.time()
    
    # Asegurar que el formato de los datos sea compatible con el modelo
    def ensure_compatible_format(dataset):
        def process_batch(images, data_dict):
            # Extraer labels y boxes del diccionario
            labels = data_dict['classification']
            boxes = data_dict['regression']
            
            # Tomar solo la primera etiqueta y caja para cada imagen en el lote
            # para compatibilidad con la arquitectura actual del modelo
            return images, {'classification': labels[:, 0], 'regression': boxes[:, 0, :]}
        
        return dataset.map(process_batch)
    
    # Aplicar la transformación
    compatible_test_dataset = ensure_compatible_format(test_dataset)
    
    # Evaluar modelo
    results = model.evaluate(compatible_test_dataset)
    
    # Mostrar resultados
    print("\nResultados de evaluación:")
    metric_names = ['loss', 'classification_loss', 'regression_loss', 
                   'classification_accuracy', 'regression_mse']
    
    for name, value in zip(metric_names, results):
        print(f"{name}: {value:.4f}")
    
    # Visualizar predicciones usando el dataset original (no transformado)
    print("\nVisualizando predicciones en conjunto de prueba...")
    
    # Crear directorio para guardar ejemplos
    os.makedirs(output_dir, exist_ok=True)
    
    # Usar directamente el dataset sin transformación extra
    visualize_predictions(model, test_dataset, 
                         class_names, readable_class_names, 
                         img_size=img_size, output_dir=output_dir)