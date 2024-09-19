import pytest
import numpy as np
from Packaging_ML_Model.packaging_ml_model.prediction_model import config
from Packaging_ML_Model.packaging_ml_model.prediction_model.processing.data_handling import load_pipeline, \
    separate_data, load_dataset
import os
from pathlib import Path
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))


print(f"Config module location: {config.__file__}")
print(f"MODEL_NAME: {config.MODEL_NAME}")


# Fixtures --> functions before test function --> ensure single_prediction
classification_pipeline = load_pipeline(config.MODEL_NAME)


@pytest.fixture
def single_prediction():
    test_data = load_dataset(config.TEST_FILE)
    X, y = separate_data(test_data)
    pred = classification_pipeline.predict(X)
    return pred


def test_single_pred_not_none(single_prediction):  # output is not none
    assert single_prediction is not None


def test_single_pred_str_type(single_prediction):  # data type is integer
    print(f"single_prediction[0]: {single_prediction[0]}, type: {type(single_prediction[0])}")
    assert isinstance(single_prediction[0], np.int64)
