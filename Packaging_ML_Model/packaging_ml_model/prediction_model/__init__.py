import os
from Packaging_ML_Model.packaging_ml_model.prediction_model.config import config

with open(os.path.join(config.PACKAGE_ROOT, 'VERSION')) as f:
    __version__ = f.read().strip()
