# Packaging the ML Model of Classification

#### Problem Statement
- Company wants to automate the loan eligibility process based on customer detail provided while filling online application form. 
- It is a classification problem where we have to predict whether a loan would be approved or not.

#### Data
The data corresponds to a set of financial requests associated with individuals. 

The loan approval dataset is a collection of financial records and associated information used to determine the eligibility of individuals or organizations for obtaining loans from a lending institution. It includes various factors such as cibil score, income, employment status, loan term, loan amount, assets value, and loan status.

Source: Kaggle


## Virtual Environment
Install virtualenv

Create virtual environment

```python
virtualenv ml_package
```

Activate virtual environment

For Windows
```python
ml_package\Scripts\activate
```

Deactivate virtual environment

```python
deactivate
```

# Directory Structure

```bash
Packaging_ML_Model/
├── Experiment/                  # (Potentially for experiment tracking or trials)
├── packaging_ml_model/          # Main folder for your ML package
│   ├── build/                   # Build-related files (when packaging)
│   ├── dist/                    # Distribution files (when packaging)
│   ├── ml_package/              # Another potential package folder (not much info here)
│   ├── prediction_model/        # Core module for prediction models
│   │   ├── config/              # Stores configuration files
│   │   │   ├── config.py        # Contains configuration settings
│   │   │   ├── __init__.py      # Makes config a module
│   │   ├── datasets/            # Stores datasets for training or testing
│   │   ├── processing/          # Contains data processing scripts
│   │   │   ├── data_handling.py # Handles loading, cleaning, etc.
│   │   │   ├── preprocessing.py # Preprocessing steps (e.g., scaling, encoding)
│   │   │   ├── __init__.py      # Makes processing a module
│   │   ├── trained_models/      # Directory for storing saved models
│   │   ├── pipeline.py          # Script defining the pipeline for training
│   │   ├── predict.py           # Script for making predictions
│   │   ├── training_pipeline.py # Script for orchestrating the training pipeline
│   │   ├── VERSION              # Stores versioning information
│   │   ├── __init__.py          # Makes prediction_model a module
├── requirements.txt             # Lists Python dependencies for the project
├── setup.py                     # Script for packaging and distribution
├── train.py                     # Entry-point for training the model
└── README.md                    # Project documentation



# Build the Package

1. Goto Project directory and install dependencies
`pip install -r requirements.txt`

2. Create Pickle file after training:
`python prediction_model/training_pipeline.py`

3. Create source distribution and wheel
`python setup.py sdist bdist_wheel`


# Installation of Package

Go to project directory where `setup.py` file is located

1. To install it in editable or developer mode
```python
pip install -e .
```
```.``` refers to current directory

```-e``` refers to --editable mode

