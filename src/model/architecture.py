"""
Definición de arquitectura del modelo para detección de objetos.
"""

import tensorflow as tf

def build_model(num_classes, input_shape=(416, 416, 3)):
    """
    Construye un modelo de detección de objetos basado en EfficientNetB3.
    
    Args:
        num_classes (int): Número de clases a detectar
        input_shape (tuple): Forma de entrada (altura, ancho, canales)
        
    Returns:
        tf.keras.Model: Modelo compilado para detección de objetos
    """
    print("\nConstruyendo modelo de detección...")
    start_time = tf.timestamp()
    
    # Usar EfficientNetB3 como modelo base
    base_model = tf.keras.applications.EfficientNetB3(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Congelar modelo base inicialmente
    base_model.trainable = False 
    # Añadir capas de detección
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Salida para clasificación
    classification_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='classification')(x)
    
    # Salida para regresión de cajas
    regression_output = tf.keras.layers.Dense(4, name='regression')(x)
    
    model = tf.keras.Model(inputs=base_model.input, outputs=[classification_output, regression_output])
    
    end_time = tf.timestamp()
    print(f"Modelo construido en {end_time - start_time:.2f} segundos.")
    return model

def configure_gpu():
    """
    Configura GPU para entrenamiento.
    
    Returns:
        list: Lista de dispositivos GPU disponibles
    """
    print("Versión de TensorFlow:", tf.__version__)
    gpus = tf.config.list_physical_devices('GPU')
    print("GPUs disponibles:", gpus)

    if gpus:
        try:
            # El crecimiento de memoria debe establecerse antes de que se inicialicen las GPU
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Crecimiento de memoria GPU habilitado")
            
            # Configuración de límite de memoria
            # Nota: 10GB se elige como un equilibrio entre los requisitos del modelo y los recursos del sistema
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=10000)])  # 10GB
            print("GPU configurada con 10GB de memoria")
            
            # Prueba de verificación de GPU
            # Realiza multiplicación de matrices para asegurar que la GPU funciona correctamente
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
                c = tf.matmul(a, b)
                print("Operación de prueba en GPU:", c)
        except RuntimeError as e:
            print("Error al configurar GPU:", e)
    else:
        print("No se detectaron GPUs, usando CPU")
    
    return gpus