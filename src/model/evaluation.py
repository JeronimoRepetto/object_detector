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
    
    # Evaluar modelo
    results = model.evaluate(test_dataset)
    
    # Mostrar resultados
    print("\nResultados de evaluación:")
    metric_names = ['loss', 'classification_loss', 'regression_loss', 
                   'classification_accuracy', 'regression_mse']
    
    for name, value in zip(metric_names, results):
        print(f"{name}: {value:.4f}")
    
    # Visualizar algunas predicciones
    print("\nVisualizando predicciones en conjunto de prueba...")
    
    # Crear directorio para guardar ejemplos
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualizar predicciones
    visualize_predictions(model, test_dataset, class_names, readable_class_names, 
                        img_size=img_size, output_dir=output_dir)