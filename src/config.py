"""
Configuración para el pipeline de detección de objetos.
Contiene rutas de datos, parámetros del modelo y configuraciones de entrenamiento.
"""

import os

# Rutas de directorios de datos
TRAIN_DIR = "/media/mturco/Storage/AI-Boost/target_dir/train"
VALIDATION_DIR = "/media/mturco/Storage/AI-Boost/target_dir/validation"
TEST_DIR = "/media/mturco/Storage/AI-Boost/target_dir/test"

# Archivos de anotaciones
TRAIN_BOXES_CSV = "/media/mturco/Storage/AI-Boost/boxes/oidv6-train-annotations-bbox.csv"
VALIDATION_BOXES_CSV = "/media/mturco/Storage/AI-Boost/boxes/validation-annotations-bbox.csv"
TEST_BOXES_CSV = "/media/mturco/Storage/AI-Boost/boxes/test-annotations-bbox.csv"
CLASS_DESCRIPTIONS_FILE = "/media/mturco/Storage/AI-Boost/boxes_Class_names/oidv7-class-descriptions-boxable.csv"

# Controles de tamaño de conjunto de datos
MAX_TRAIN_ANNOTATIONS = None
MAX_VAL_ANNOTATIONS = None
MAX_TEST_ANNOTATIONS = None

# Configuración del modelo
IMG_SIZE = (416, 416)
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001

# Directorios de salida
OUTPUT_DIR = os.path.join(os.getcwd(), "output")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
EXAMPLES_DIR = os.path.join(OUTPUT_DIR, "examples")
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")
DEPLOYMENT_DIR = os.path.join(OUTPUT_DIR, "deployment_model")

# Crear directorios si no existen
for directory in [OUTPUT_DIR, MODELS_DIR, LOGS_DIR, EXAMPLES_DIR, PREDICTIONS_DIR, DEPLOYMENT_DIR]:
    os.makedirs(directory, exist_ok=True)

# Configuración GPU
GPU_MEMORY_LIMIT = 10000  # 10GB