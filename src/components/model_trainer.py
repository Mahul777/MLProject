# Trains ML model

# 🚀 Training the Machine Learning Model
# It handles the entire logic of selecting, training, evaluating, and
#  saving the best-performing machine learning model from a list of candidates.

import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models
import pandas as pd  # Required for data cleaning


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # ✅ CLEAN the y_train and y_test
            y_train = pd.Series(y_train).astype(str).str.replace('"', '').str.strip().astype(float)
            y_test = pd.Series(y_test).astype(str).str.replace('"', '').str.strip().astype(float)

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [.1, .01, 0.5, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                models=models, param=params
            )

            # Get best model score from the report
            best_model_score = max(sorted(model_report.values()))

            # Get best model name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No suitable model found with R2 > 0.6")

            logging.info(f"Best model: {best_model_name} with R2 Score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)

# you are building the model training and selection part of your machine learning pipeline. 
# Here’s what you are trying to do:

# Define where to save the trained model:
# The ModelTrainerConfig dataclass sets the file path for saving the best model.

# ModelTrainer class:

# Splits the input arrays into features (X_train, X_test) and targets (y_train, y_test).
# Cleans the target variables to ensure they are numeric and free of unwanted characters.
# Defines several regression models (Random Forest, Decision Tree, Gradient Boosting, etc.) and their 
# hyperparameters for tuning.
# Evaluates all models using grid search (evaluate_models), which finds the best hyperparameters for 
# each model and computes their R² scores.
# Selects the best model based on the highest R² score.
# Checks if the best model’s R² score is above 0.6; if not, it raises an exception.
# Saves the best model to disk for later use.
# Returns the R² score of the best model on the test set.
# Summary:
# This code automates the process of training, tuning, evaluating, selecting, and saving the 
# best regression model for your dataset.

