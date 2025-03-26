"""
Funciones para preprocesar datos y crear conjuntos de datos TensorFlow.
"""

import tensorflow as tf
import numpy as np
import os
import time

def create_dataset(image_dir, annotations_dict):
    """
    Crea un conjunto de datos TensorFlow a partir de un directorio de imágenes y un diccionario de anotaciones.
    
    Args:
        image_dir (str): Directorio que contiene imágenes
        annotations_dict (dict): Diccionario que mapea nombres de archivo a anotaciones
        
    Returns:
        tf.data.Dataset o None: Conjunto de datos que contiene tuplas (filename, boxes, labels),
                               o None si ocurrió un error
    """
    print(f"Creando conjunto de datos desde {image_dir}...")
    start_time = time.time()
    
    filenames = []
    boxes_list = []
    labels_list = []
    
    # Verificar si el directorio existe
    if not os.path.exists(image_dir):
        print(f"Error: El directorio {image_dir} no existe.")
        return None
    
    # Iterar sobre las anotaciones
    for filename, annotations in annotations_dict.items():
        # Construir ruta completa de la imagen
        img_path = os.path.join(image_dir, filename)
        
        # Verificar si la imagen existe
        if not os.path.exists(img_path):
            continue
        
        # Extraer cajas y etiquetas
        boxes = []
        labels = []
        
        for box, label in annotations:
            boxes.append(box)
            labels.append(label)
        
        # Añadir a las listas
        filenames.append(img_path)
        
        # Convertir a arrays NumPy con padding
        max_boxes = 100  # Número máximo de cajas por imagen
        padded_boxes = np.zeros((max_boxes, 4), dtype=np.float32)
        padded_labels = np.zeros(max_boxes, dtype=np.int32)
        
        # Rellenar con valores reales
        num_boxes = min(len(boxes), max_boxes)
        padded_boxes[:num_boxes] = np.array(boxes[:num_boxes], dtype=np.float32)
        padded_labels[:num_boxes] = np.array(labels[:num_boxes], dtype=np.int32)
        
        boxes_list.append(padded_boxes)
        labels_list.append(padded_labels)
    
    if not filenames:
        print(f"Error: No se encontraron imágenes en {image_dir} que coincidan con las anotaciones.")
        return None
    
    # Convertir listas a tensores
    boxes_tensor = np.array(boxes_list)
    labels_tensor = np.array(labels_list)
    
    # Crear conjunto de datos
    dataset = tf.data.Dataset.from_tensor_slices((filenames, boxes_tensor, labels_tensor))
    
    print(f"Conjunto de datos creado en {time.time() - start_time:.2f} segundos. {len(filenames)} imágenes incluidas.")
    return dataset

def load_image(img_path, boxes, labels):
    """
    Carga una imagen y sus anotaciones.
    
    Args:
        img_path (str): Ruta al archivo de imagen
        boxes (tensor): Tensor de cajas delimitadoras
        labels (tensor): Tensor de etiquetas
        
    Returns:
        tuple: (image, valid_boxes, valid_labels)
            - image: Tensor de imagen decodificada
            - valid_boxes: Tensor que contiene solo cajas válidas (sin padding)
            - valid_labels: Tensor que contiene solo etiquetas válidas (sin padding)
    """
    # Leer imagen
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    
    # Encontrar cajas y etiquetas válidas (sin padding)
    valid_indices = tf.where(tf.reduce_sum(boxes, axis=1) > 0)
    valid_boxes = tf.gather(boxes, valid_indices)
    valid_labels = tf.gather(labels, valid_indices)
    
    # Aplanar índices
    valid_boxes = tf.reshape(valid_boxes, [-1, 4])
    valid_labels = tf.reshape(valid_labels, [-1])
    
    return img, valid_boxes, valid_labels

def preprocess_dataset(dataset, img_size, batch_size, augment=False, is_training=True):
    """
    Preprocesa un conjunto de datos para entrenamiento o visualización.
    
    Args:
        dataset (tf.data.Dataset): Conjunto de datos a preprocesar
        img_size (tuple): Tamaño objetivo de imagen (altura, ancho)
        batch_size (int): Tamaño de lote
        augment (bool): Si se debe aplicar aumento de datos
        is_training (bool): Si el conjunto de datos es para entrenamiento
        
    Returns:
        tf.data.Dataset: Conjunto de datos preprocesado
    """
    def load_and_preprocess_image(img_path, boxes, labels):
        # Cargar imagen
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        
        # Redimensionar imagen
        img = tf.image.resize(img, img_size)
        
        # Normalizar imagen
        img = img / 255.0
        
        # Aplicar aumento de datos si es necesario
        if augment:
            # Volteo horizontal aleatorio
            if tf.random.uniform(()) > 0.5:
                img = tf.image.flip_left_right(img)
                # También necesitamos ajustar las coordenadas de las cajas
                flipped_boxes = tf.stack([
                    boxes[:, 0],           # y_min permanece igual
                    1.0 - boxes[:, 3],     # el nuevo x_min es 1 - x_max original
                    boxes[:, 2],           # y_max permanece igual
                    1.0 - boxes[:, 1]      # el nuevo x_max es 1 - x_min original
                ], axis=1)
                boxes = flipped_boxes
            
            # Brillo aleatorio
            img = tf.image.random_brightness(img, 0.1)
            
            # Contraste aleatorio
            img = tf.image.random_contrast(img, 0.9, 1.1)
            
            # Saturación aleatoria
            img = tf.image.random_saturation(img, 0.9, 1.1)
            
            # Recortar aleatoriamente
            img = tf.clip_by_value(img, 0.0, 1.0)
        
        # Filtrar cajas vacías
        valid_indices = tf.reduce_sum(boxes, axis=1) > 0
        filtered_boxes = tf.boolean_mask(boxes, valid_indices)
        filtered_labels = tf.boolean_mask(labels, valid_indices)
        
        # Limitar a 25 cajas por imagen para mantener un tamaño constante
        max_boxes = 25
        num_boxes = tf.minimum(tf.shape(filtered_boxes)[0], max_boxes)
        
        # Tomar las primeras max_boxes cajas y etiquetas
        final_boxes = filtered_boxes[:num_boxes]
        final_labels = filtered_labels[:num_boxes]
        
        # Rellenar con ceros hasta llegar a max_boxes
        padded_boxes = tf.pad(
            final_boxes, 
            [[0, max_boxes - num_boxes], [0, 0]], 
            constant_values=0.0
        )
        padded_labels = tf.pad(
            final_labels, 
            [[0, max_boxes - num_boxes]], 
            constant_values=0
        )
        
        return img, padded_boxes, padded_labels
    
    # Mapear función de carga y preprocesamiento
    dataset = dataset.map(
        load_and_preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Si es para entrenamiento, transformar la salida para coincida con la estructura del modelo
    if is_training:
        dataset = dataset.map(
            lambda img, boxes, labels: (img, {'classification': labels, 'regression': boxes}),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    # Configurar comportamiento del conjunto de datos
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def verify_dataset(dataset, class_names):
    """
    Verifica que un conjunto de datos tenga el formato correcto.
    
    Args:
        dataset (tf.data.Dataset): Conjunto de datos a verificar
        class_names (list): Lista de nombres de clases
        
    Returns:
        bool: True si la verificación pasó, False en caso contrario
    """
    try:
        # Obtener un lote
        for images, boxes, labels in dataset.take(1):
            print(f"Forma de imagen: {images.shape}")
            print(f"Forma de cajas: {boxes.shape}")
            print(f"Forma de etiquetas: {labels.shape}")
            
            # Verificar rango de imagen
            print(f"Rango de valores de imagen: [{tf.reduce_min(images)}, {tf.reduce_max(images)}]")
            
            # Verificar rango de cajas
            print(f"Rango de valores de cajas: [{tf.reduce_min(boxes)}, {tf.reduce_max(boxes)}]")
            
            # Verificar etiquetas
            unique_labels = tf.unique(tf.reshape(labels, [-1]))[0]
            print(f"Etiquetas únicas en el lote: {unique_labels.numpy()}")
            
            # Verificar que las etiquetas estén dentro del rango válido
            max_label = tf.reduce_max(labels)
            if max_label >= len(class_names):
                print(f"¡Advertencia! Etiqueta máxima ({max_label}) fuera del rango de clases ({len(class_names)}).")
            
            return True
    except Exception as e:
        print(f"Error al verificar el conjunto de datos: {e}")
        return False