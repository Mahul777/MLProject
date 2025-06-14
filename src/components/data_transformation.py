# Transforms features
#✅ The Data Transformation file takes care of cleaning, transforming, and preparing the dataset
#  so that your model can understand and learn from it effectively.

import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import os

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

# ✅ Configuration class to define the preprocessor file path
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")


# ✅ Main class for data transformation logic
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        Creates preprocessing pipelines for both numerical and categorical columns.
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # 🔢 Numerical pipeline
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            # 🧠 Categorical pipeline
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))
            ])

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # 🔗 Combine pipelines
            preprocessor = ColumnTransformer(transformers=[
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        '''
        Reads datasets, cleans data, applies transformation, and returns arrays + preprocessor path.
        '''
        try:
            # 📥 Load data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # 🧹 Clean column names
            train_df.columns = (
                train_df.columns.str.strip()
                .str.lower()
                .str.replace(' ', '_')
                .str.replace('"', '')
            )
            test_df.columns = (
                test_df.columns.str.strip()
                .str.lower()
                .str.replace(' ', '_')
                .str.replace('"', '')
            )

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            # 🎯 Target and numeric columns
            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # ✅ Ensure target column is present
            assert target_column_name in train_df.columns, \
                f"Target column '{target_column_name}' not found in train data."
            assert target_column_name in test_df.columns, \
                f"Target column '{target_column_name}' not found in test data."

            # ✅ Clean numeric columns to remove quotes and convert to float
            for col in numerical_columns:
                train_df[col] = (
                    train_df[col]
                    .astype(str)
                    .str.replace('"', '')
                    .str.strip()
                    .astype(float)
                )
                test_df[col] = (
                    test_df[col]
                    .astype(str)
                    .str.replace('"', '')
                    .str.strip()
                    .astype(float)
                )

            # ➗ Split input features and target
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing on train and test data")

            # 🛠 Get and apply preprocessor
            preprocessing_obj = self.get_data_transformer_object()

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # 🧱 Combine features + target
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # 💾 Save preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Preprocessing object saved successfully")

            # ✅ Return results
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)





# you are building a data transformation pipeline for a machine learning project. Here’s what you are trying to do:

# Set up configuration:
# You define where to save the preprocessor object (a file that stores your data transformation steps).

# Create a transformation class:
# The DataTransformation class handles all data preprocessing.

# Build preprocessing pipelines:

# For numerical columns (writing_score, reading_score):
# Fill missing values with the median.
# Standardize (scale) the values.
# For categorical columns (like gender, lunch, etc.):
# Fill missing values with the most frequent value.
# Convert categories to numbers using one-hot encoding.
# Scale the encoded values.
# Apply transformations:

# Read the training and test datasets.
# Separate features and target (math_score).
# Fit the preprocessing pipeline on the training data and transform both train and test data.
# Combine the transformed features with the target variable.
# Save the preprocessor:

# Store the fitted preprocessor object to disk so you can use the same transformations later (e.g., during model deployment).
# In summary:
# You are automating the process of cleaning, encoding, and scaling your data, and saving the transformation steps for consistent use in your ML workflow.

