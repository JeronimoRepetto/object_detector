#!/usr/bin/env python3
"""
Script principal para entrenar el modelo de detección de objetos.
"""
from src.config import *
from src.data import (
    check_csv_format, load_annotations, extract_class_names_from_annotations,
    convert_open_images_to_standard_format, create_annotations_dict,
    load_class_descriptions, create_dataset, preprocess_dataset,
    verify_dataset, visualize_examples
)
from src.model import build_model, configure_gpu, train_model, evaluate_model
from src.deployment.exporter import export_saved_model
import os

def main():
    """Función principal del pipeline de detección de objetos."""
    # Configurar GPU
    gpus = configure_gpu()
    
    # Verificar archivos CSV
    print("\n1. Verificando formato de archivos CSV...")
    check_csv_format(TRAIN_BOXES_CSV)
    check_csv_format(VALIDATION_BOXES_CSV)
    check_csv_format(TEST_BOXES_CSV)
    
    # Cargar anotaciones
    print("\n2. Cargando anotaciones...")
    train_df = load_annotations(TRAIN_BOXES_CSV, MAX_TRAIN_ANNOTATIONS)
    val_df = load_annotations(VALIDATION_BOXES_CSV, MAX_VAL_ANNOTATIONS)
    test_df = load_annotations(TEST_BOXES_CSV, MAX_TEST_ANNOTATIONS)
    
    # Extraer nombres de clases
    print("\n3. Extrayendo nombres de clases...")
    class_names = extract_class_names_from_annotations(train_df, val_df, test_df)
    num_classes = len(class_names)
    print(f"Número total de clases: {num_classes}")
    
    # Cargar descripciones de clases
    print("\n4. Cargando descripciones de clases...")
    class_descriptions = load_class_descriptions(CLASS_DESCRIPTIONS_FILE)
    
    # Mostrar las primeras 10 clases y sus descripciones para verificar
    print("\nVerificando mapeo de clases a descripciones:")
    print("ID de Clase | Descripción")
    print("-" * 50)
    for i, class_id in enumerate(class_names[:10]):
        description = class_descriptions.get(class_id, "No disponible")
        print(f"{i}: {class_id} | {description}")
    
    # Convertir a formato estándar
    print("\n5. Convirtiendo anotaciones a formato estándar...")
    train_std = convert_open_images_to_standard_format(train_df)
    val_std = convert_open_images_to_standard_format(val_df)
    test_std = convert_open_images_to_standard_format(test_df)
    
    # Crear diccionarios de anotaciones
    print("\n6. Creando diccionarios de anotaciones...")
    train_annotations, label_encoder = create_annotations_dict(train_std, class_names)
    val_annotations, _ = create_annotations_dict(val_std, class_names)
    test_annotations, _ = create_annotations_dict(test_std, class_names)
    
    # Guardar mapeo de índices a nombres para referencia
    print("\nGuardando mapeo de índices a nombres...")
    with open(os.path.join(OUTPUT_DIR, "class_mapping.txt"), "w") as f:
        f.write("Índice,ID de Clase,Nombre Legible\n")
        for idx, class_id in enumerate(label_encoder.classes_):
            readable_name = class_descriptions.get(class_id, class_id)
            f.write(f"{idx},{class_id},{readable_name}\n")
    
    # Mostrar la codificación de algunas clases
    print("\nVerificando la codificación de clases:")
    print("Índice | ID de Clase")
    print("-" * 30)
    for i, class_id in enumerate(label_encoder.classes_[:10]):
        print(f"{i}: {class_id}")
    
    # Crear conjuntos de datos
    print("\n7. Creando conjuntos de datos TensorFlow...")
    train_ds = create_dataset(TRAIN_DIR, train_annotations)
    val_ds = create_dataset(VALIDATION_DIR, val_annotations)
    test_ds = create_dataset(TEST_DIR, test_annotations)
    
    # Preprocesar conjuntos de datos
    print("\n8. Preprocesando conjuntos de datos...")
    # Dataset de verificación (mantiene formato original)
    verify_ds = preprocess_dataset(train_ds, IMG_SIZE, BATCH_SIZE, augment=False, is_training=False)

    # Datasets para entrenamiento (formato adaptado para el modelo)
    train_ds = preprocess_dataset(train_ds, IMG_SIZE, BATCH_SIZE, augment=True, is_training=True)
    val_ds = preprocess_dataset(val_ds, IMG_SIZE, BATCH_SIZE, augment=False, is_training=True)
    test_ds = preprocess_dataset(test_ds, IMG_SIZE, BATCH_SIZE, augment=False, is_training=True)
    
    # Verificar conjuntos de datos
    print("\n9. Verificando conjuntos de datos...")
    verify_dataset(verify_ds, class_names)
    
    # Crear lista de nombres legibles para cada clase SEGÚN EL ORDEN DEL ENCODER
    print("\n10. Preparando nombres legibles de clases...")
    readable_class_names = []
    for class_id in label_encoder.classes_:
        readable_name = class_descriptions.get(class_id, class_id)
        readable_class_names.append(readable_name)
    
    print(f"Nombres legibles (primeros 5): {readable_class_names[:5]}")
    
    # Visualizar ejemplos
    print("\n11. Visualizando ejemplos de entrenamiento...")
    visualize_examples(
        verify_ds, 
        label_encoder.classes_,  # Lista ordenada de IDs de clase tal como los conoce el encoder
        num_examples=5, 
        img_size=IMG_SIZE, 
        class_descriptions=class_descriptions,
        output_dir=EXAMPLES_DIR
    )
    
    # Construir modelo
    print("\n12. Construyendo modelo...")
    model = build_model(num_classes, input_shape=IMG_SIZE + (3,))
    
    # Entrenar modelo
    print("\n13. Entrenando modelo...")
    history = train_model(
        model,
        train_ds,
        val_ds,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        models_dir=MODELS_DIR,
        logs_dir=LOGS_DIR
    )
    
    # Guardar o visualizar el historial
    print(f"Resultados del entrenamiento: {history.history}")
    
    # Evaluar modelo
    print("\n14. Evaluando modelo...")
    evaluate_model(
        model,
        test_ds,
        label_encoder.classes_,  # Usar directamente la lista ordenada del encoder
        readable_class_names=readable_class_names,
        img_size=IMG_SIZE,
        output_dir=PREDICTIONS_DIR
    )
    
    # Exportar modelo
    print("\n15. Exportando modelo para despliegue...")
    export_saved_model(model, DEPLOYMENT_DIR)
    
    print("\n¡Pipeline completado con éxito!")

if __name__ == "__main__":
    main()