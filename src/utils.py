# Common helper functions
#utils.py file is used to store common utility functions that are reused across 
# different parts of the machine learning project.


import os
import sys

import numpy as np 
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

# Function to save a Python object to disk using dill
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        # Create the directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)
        # Open the file in write-binary mode and dump the object
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        # Raise a custom exception if something goes wrong
        raise CustomException(e, sys)
    
# Function to evaluate multiple models using GridSearchCV and return their R2 scores
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}  # Dictionary to store model names and their test R2 scores

        # Loop through each model
        for i in range(len(list(models))):
            model = list(models.values())[i]  # Get the model object
            para = param[list(models.keys())[i]]  # Get the hyperparameters for this model

            # Perform grid search cross-validation to find best hyperparameters
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            # Set the model to the best found parameters and fit on training data
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Predict on train and test sets
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate R2 scores for train and test sets
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store the test R2 score in the report dictionary
            report[list(models.keys())[i]] = test_model_score

        return report  # Return the dictionary of model names and their test R2 scores

    except Exception as e:
        # Raise a custom exception if something goes wrong
        raise CustomException(e, sys)
    
# Function to load a Python object from disk using dill
def load_object(file_path):
    try:
        # Open the file in read-binary mode and load the object
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        # Raise a custom exception if something goes wrong
        raise CustomException(e, sys)

# you are creating utility functions to help with your machine learning workflow:

# save_object:
# Saves any Python object (like a model or preprocessor) to disk using dill, so you can reuse it later.

# evaluate_models:
# Takes several models and their hyperparameters, performs grid search (GridSearchCV) to find the best parameters for each model,
#  trains them, and evaluates their performance using the R² score on the test set. It returns a report (dictionary) of model names 
# and their test R² scores.

# load_object:
# Loads a previously saved Python object (like a model or preprocessor) from disk using dill.

# Summary:
# These functions help you save and load objects, and automate the process of training, tuning, and evaluating multiple machine 
# learning models. This makes your ML pipeline more organized and reusable.

