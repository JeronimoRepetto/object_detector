"""
Funciones para entrenar el modelo de detección de objetos.
"""

import tensorflow as tf
import pandas as pd
import os 
import time
from tensorflow.keras.mixed_precision import set_global_policy

def train_model(model, train_dataset, val_dataset, epochs=10, learning_rate=0.001,
               models_dir='models', logs_dir='logs'):
    """
    Entrena el modelo con los conjuntos de datos proporcionados.
    
    Args:
        model (tf.keras.Model): Modelo a entrenar
        train_dataset (tf.data.Dataset): Conjunto de datos de entrenamiento
        val_dataset (tf.data.Dataset): Conjunto de datos de validación
        epochs (int): Número de épocas de entrenamiento
        learning_rate (float): Tasa de aprendizaje
        models_dir (str): Directorio para guardar modelos
        logs_dir (str): Directorio para guardar registros
        
    Returns:
        tf.keras.callbacks.History: Historial de entrenamiento
    """
    print("\nIniciando entrenamiento del modelo...")
    start_time = time.time()
    
    # Configurar la política de precisión mixta para acelerar el entrenamiento
    set_global_policy('mixed_float16')
    print("Usando precisión mixta para el entrenamiento")
    
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
    train_dataset = ensure_compatible_format(train_dataset)
    val_dataset = ensure_compatible_format(val_dataset)
    
    # Compilar modelo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            'classification': 'sparse_categorical_crossentropy',
            'regression': 'mse'
        },
        metrics={
            'classification': 'accuracy',
            'regression': 'mse'
        }
    )
    
    # Crear directorio para guardar modelos
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Callbacks para entrenamiento
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(models_dir, 'best_model.h5'),
            save_best_only=True,
            monitor='val_classification_accuracy',
            mode='max'
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=3,
            restore_best_weights=True,
            monitor='val_classification_accuracy',
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.2,
            patience=2,
            min_lr=0.00001,
            monitor='val_classification_accuracy',
            mode='max'
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=logs_dir,
            histogram_freq=1
        )
    ]
    
    # Entrenar modelo
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks
    )
    
    # Guardar modelo final
    model.save(os.path.join(models_dir, 'final_model.h5'))
    
    # Guardar historial de entrenamiento
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(os.path.join(models_dir, 'training_history.csv'), index=False)
    
    training_time = time.time() - start_time
    print(f"\nEntrenamiento completado en {training_time:.2f} segundos ({training_time/60:.2f} minutos).")
    
    return history