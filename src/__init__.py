# Autonomous Drone Detection Package
__version__ = "1.0.0"
__author__ = "Autonomous Drone Detection Team"

from . import data_processing
from . import models
from . import training
from . import inference
from . import evaluation
from . import utils

__all__ = [
    'data_processing',
    'models', 
    'training',
    'inference',
    'evaluation',
    'utils'
]