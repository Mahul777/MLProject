# Reads and splits data
# The file data_ingestion.py is used for the first and most important step in a machine learning project pipeline:
# ✅ Loading, saving, and preparing your raw data for further processing.

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# ✅ Fix: Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.exception import CustomException
from src.logger import logging

# 📦 Configuration class for file paths
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")


# 📥 Main ingestion class
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")

        try:
            # Load data
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the dataset as dataframe')

            # Create directories if not exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Train-test split
            logging.info("Train-test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save splits
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


# 🔁 Entry point
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()


#your code is doing, step by step
# Imports:
# Loads necessary libraries for file handling, data processing, machine learning, and custom error/logging.

# Configuration:
# Uses a dataclass (DataIngestionConfig) to define and store file paths for saving the raw, training, and test datasets.

# DataIngestion Class:

# Handles the entire data ingestion process.
# Reads the raw dataset (stud.csv) from your notebook’s data folder.
# Ensures the output directory (artifacts/) exists.
# Saves the raw data to a specified location.
# Splits the data into training and test sets (80%/20%).
# Saves these sets to their respective file paths.
# Logs each step for traceability.
# Handles errors using a custom exception.
# Execution:
# If you run the script directly, it creates a DataIngestion object and starts the ingestion process.

    




