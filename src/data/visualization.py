"""
Funciones para visualizar ejemplos de datos y predicciones del modelo.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import tensorflow as tf

def visualize_examples(dataset, class_names, num_examples=5, img_size=(416, 416), 
                     class_descriptions=None, output_dir='examples'):
    """
    Visualiza ejemplos del conjunto de datos con sus cajas delimitadoras.
    
    Args:
        dataset (tf.data.Dataset): Conjunto de datos a visualizar
        class_names (list): Lista de IDs de clases (ej. /m/0mkg)
        num_examples (int): Número de ejemplos a visualizar
        img_size (tuple): Tamaño de imagen (altura, ancho)
        class_descriptions (dict): Diccionario que mapea IDs de clases a nombres legibles
        output_dir (str): Directorio donde guardar los ejemplos
    """
    # Crear directorio para guardar ejemplos
    os.makedirs(output_dir, exist_ok=True)
    
    # Colores para diferentes clases
    colors = plt.cm.hsv(np.linspace(0, 1, len(class_names)))
    
    # Tomar algunos ejemplos - mezclar primero para obtener ejemplos diferentes cada vez
    examples = list(dataset.unbatch().shuffle(buffer_size=1000).take(num_examples))
    
    print(f"\nProcesando {len(examples)} ejemplos para visualización")
    
    for i, (image, boxes, labels) in enumerate(examples):
        # Convertir a NumPy
        image_np = image.numpy()
        boxes_np = boxes.numpy()
        labels_np = labels.numpy()
        
        # Información de depuración
        print(f"\nEjemplo {i+1}:")
        print(f"Forma de la imagen: {image_np.shape}")
        print(f"Número de cajas: {len(boxes_np)}")
        print(f"Etiquetas (índices): {labels_np}")
        
        # Crear figura
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(image_np)
        
        # Dibujar cajas
        for j, (box, label_idx) in enumerate(zip(boxes_np, labels_np)):
            # Obtener coordenadas
            y_min, x_min, y_max, x_max = box
            
            # Ignorar cajas con coordenadas inválidas (pueden ser padding)
            if np.sum(box) == 0:
                continue
            
            # Convertir a píxeles
            x_min_px = x_min * img_size[1]
            y_min_px = y_min * img_size[0]
            width_px = (x_max - x_min) * img_size[1]
            height_px = (y_max - y_min) * img_size[0]
            
            # Obtener color para la clase - asegurar que sea entero
            label_idx_int = int(label_idx)
            color = colors[label_idx_int % len(colors)]
            
            # Dibujar rectángulo
            rect = patches.Rectangle(
                (x_min_px, y_min_px), width_px, height_px,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # Añadir etiqueta - usar nombre legible si está disponible
            try:
                # Verificar que el índice está dentro del rango
                if label_idx_int < len(class_names):
                    # Obtener el ID de clase usando el índice
                    class_id = class_names[label_idx_int]
                    
                    # Información de depuración
                    print(f"  Caja {j}: Índice {label_idx_int} -> ID {class_id}")
                    
                    # Obtener el nombre legible usando el ID de clase
                    if class_descriptions and class_id in class_descriptions:
                        display_name = class_descriptions[class_id]
                        print(f"    Descripción: {display_name}")
                    else:
                        display_name = class_id
                        print(f"    No se encontró descripción para {class_id}")
                else:
                    display_name = f"Desconocido ({label_idx_int})"
                    print(f"  Caja {j}: Etiqueta {label_idx_int} fuera de rango (max: {len(class_names)-1})")
            except Exception as e:
                display_name = f"Error: {str(e)}"
                print(f"  Error al procesar etiqueta {label_idx}: {str(e)}")
            
            plt.text(
                x_min_px, y_min_px - 5,
                display_name,
                bbox=dict(facecolor=color, alpha=0.5),
                fontsize=8, color='white'
            )
        
        # Guardar figura
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"ejemplo_{i+1}.png"))
        plt.close()
    
    print(f"\nEjemplos guardados en el directorio '{output_dir}'")

def visualize_predictions(model, dataset, class_names, readable_class_names, 
                         img_size=(416, 416), output_dir='predictions'):
    """
    Visualiza las predicciones del modelo en un conjunto de datos.
    
    Args:
        model (tf.keras.Model): Modelo entrenado
        dataset (tf.data.Dataset): Conjunto de datos a visualizar
        class_names (list): Lista de IDs de clases
        readable_class_names (list): Lista de nombres de clases legibles
        img_size (tuple): Tamaño de imagen (altura, ancho)
        output_dir (str): Directorio para guardar imágenes
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Tomar un batch del dataset
    for batch in dataset.take(1):
        # Extraer imágenes y etiquetas del batch
        if isinstance(batch, tuple) and len(batch) == 2:
            # Formato: (images, data_dict)
            images, data_dict = batch
            true_boxes = data_dict['regression']
            true_labels = data_dict['classification']
        elif isinstance(batch, tuple) and len(batch) == 3:
            # Formato: (images, boxes, labels)
            images, true_boxes, true_labels = batch
        else:
            print(f"Formato de datos no reconocido. Tipo de batch: {type(batch)}")
            continue
        
        # Realizar predicciones
        predictions = model.predict(images)
        if isinstance(predictions, dict):
            pred_labels = predictions['classification']
            pred_boxes = predictions['regression']
        elif isinstance(predictions, list) and len(predictions) == 2:
            pred_labels, pred_boxes = predictions
        else:
            print(f"Formato de predicciones no reconocido. Tipo: {type(predictions)}")
            continue
        
        # Convertir one-hot a índices de clases
        if len(true_labels.shape) > 2:  # Si es one-hot
            true_labels = tf.argmax(true_labels, axis=-1)
        
        if len(pred_labels.shape) > 2:  # Si es one-hot
            pred_labels = tf.argmax(pred_labels, axis=-1)
        
        # Visualizar cada imagen del batch
        for i in range(min(5, len(images))):
            plt.figure(figsize=(10, 10))
            
            # Mostrar imagen
            img = images[i].numpy()
            plt.imshow(img)
            
            # Obtener cajas y etiquetas verdaderas
            valid_indices = tf.where(tf.reduce_sum(true_boxes[i], axis=1) > 0)
            valid_boxes = tf.gather(true_boxes[i], valid_indices).numpy()
            valid_labels = tf.gather(true_labels[i], valid_indices).numpy()
            
            # Dibujar cajas verdaderas (en verde)
            for box, label_idx in zip(valid_boxes, valid_labels):
                # Extraer coordenadas de la caja
                if len(box.shape) > 1:
                    box = box[0]  # Si hay una dimensión extra, tomar el primer elemento
                
                if len(box) == 4:  # Asegurarnos de que tengamos los 4 valores de coordenadas
                    y_min, x_min, y_max, x_max = box
                    width = x_max - x_min
                    height = y_max - y_min
                    
                    # Crear rectángulo
                    rect = patches.Rectangle(
                        (x_min * img_size[1], y_min * img_size[0]),
                        width * img_size[1],
                        height * img_size[0],
                        linewidth=2,
                        edgecolor='g', 
                        facecolor='none'
                    )
                    plt.gca().add_patch(rect)
                    
                    # Añadir etiqueta
                    label_idx_int = int(label_idx)
                    class_name = readable_class_names[label_idx_int]
                    plt.text(
                        x_min * img_size[1],
                        y_min * img_size[0] - 5,
                        f"Real: {class_name}",
                        color='g',
                        fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.7)
                    )
            
            # Obtener etiqueta predicha con mayor confianza
            pred_label_idx = np.argmax(pred_labels[i])
            pred_box = pred_boxes[i]
            
            # Verificar el formato de la caja predicha
            if len(pred_box.shape) > 0 and len(pred_box) == 4:  # Verificar que tengamos 4 valores
                y_min, x_min, y_max, x_max = pred_box
                width = x_max - x_min
                height = y_max - y_min
                
                # Crear rectángulo solo si las coordenadas son válidas
                if width > 0 and height > 0:
                    rect = patches.Rectangle(
                        (x_min * img_size[1], y_min * img_size[0]),
                        width * img_size[1],
                        height * img_size[0],
                        linewidth=2,
                        edgecolor='r',
                        facecolor='none'
                    )
                    plt.gca().add_patch(rect)
                    
                    # Añadir etiqueta
                    class_name = readable_class_names[pred_label_idx]
                    confidence = pred_labels[i][pred_label_idx]
                    plt.text(
                        x_min * img_size[1],
                        y_min * img_size[0] - 20,
                        f"Pred: {class_name} ({confidence:.2f})",
                        color='r',
                        fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.7)
                    )
            else:
                print(f"Formato de caja de predicción inesperado: {pred_box.shape}")
            
            plt.title(f"Predicción {i+1}")
            plt.axis('off')
            
            # Guardar imagen
            plt.savefig(os.path.join(output_dir, f"prediccion_{i+1}.png"))
            plt.close()
        
        print(f"Predicciones guardadas en el directorio '{output_dir}'")