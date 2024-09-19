import os
from Packaging_ML_Model.packaging_ml_model.prediction_model.processing.data_handling import load_dataset, save_pipeline, separate_data, split_data
from Packaging_ML_Model.packaging_ml_model.prediction_model.config import config
from Packaging_ML_Model.packaging_ml_model.prediction_model import pipeline as pipe
import sys
from pathlib import Path

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))


def perform_training():
    dataset = load_dataset(config.FILE_NAME)
    X, y = separate_data(dataset)
    y = y.apply(lambda x: 1 if x.strip() == "Approved" else 0)
    X_train, X_test, y_train, y_test = split_data(X, y)
    test_data = X_test.copy()
    test_data[config.TARGET] = y_test
    test_data.to_csv(os.path.join(config.DATAPATH, config.TEST_FILE))
    pipe.classification_pipeline.fit(X_train, y_train)
    save_pipeline(pipe.classification_pipeline)


if __name__ == '__main__':
    perform_training()
