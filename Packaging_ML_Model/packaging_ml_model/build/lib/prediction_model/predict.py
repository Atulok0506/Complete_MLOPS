import numpy as np
from Packaging_ML_Model.packaging_ml_model.prediction_model.config import config
from Packaging_ML_Model.packaging_ml_model.prediction_model.processing.data_handling import load_dataset, load_pipeline, separate_data
import sys
from pathlib import Path
import os
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))
classification_pipeline = load_pipeline(config.MODEL_NAME)


def generate_predictions():
    test_data = load_dataset(config.TEST_FILE)
    X, y = separate_data(test_data)
    pred = classification_pipeline.predict(X)
    output = np.where(pred == 1, 'Approved', 'Not Approved')
    print(output)
    return output


if __name__ == '__main__':
    generate_predictions()
