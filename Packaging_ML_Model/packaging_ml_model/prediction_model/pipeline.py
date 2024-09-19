from sklearn.pipeline import Pipeline
from Packaging_ML_Model.packaging_ml_model.prediction_model.config import config
from Packaging_ML_Model.packaging_ml_model.prediction_model.processing import preprocessing as pp
from sklearn.linear_model import LogisticRegression
import sys
from pathlib import Path
import os

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))


classification_pipeline = Pipeline(
    [
        ('DomainProcessing', pp.DomainProcessing(variable_to_add=config.FEATURE_TO_ADD)),
        ('DropFeatures', pp.DropColumns(variables_to_drop=config.DROP_FEATURES)),
        ('LabelEncoder', pp.CustomLabelEncoder(variables=config.FEATURES_TO_ENCODE)),
        ('LogTransform', pp.LogTransforms(variables=config.LOG_FEATURES)),
        ('LogisticClassifier', LogisticRegression(random_state=0, max_iter=1000))
    ]
)
