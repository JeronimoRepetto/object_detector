"""
Módulos para la definición, entrenamiento y evaluación de modelos.
"""

from .architecture import build_model, configure_gpu
from .training import train_model
from .evaluation import evaluate_model