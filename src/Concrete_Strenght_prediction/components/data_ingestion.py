import pandas as pd 
import numpy as np
import os
import sys

from Concrete_Strenght_prediction.logger import logging
from Concrete_Strenght_prediction.exception import customexception
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngistionConfig:
    rawdata_path: str = os.path.join("Artifacts", "raw.csv")
    traindata_path: str = os.path.join("Artifacts", "traindata.csv")
    testdata_path: str = os.path.join("Artifacts", "testdata.csv")

class DataIngistion:
    def __init__(self):
        self.DataIngistion_config = DataIngistionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Started")

        try:
            # Load the data
            data_path = Path(os.path.join("notebooks/data", "concrete_data.csv"))
            if not data_path.exists():
                logging.error(f"Data file not found: {data_path}")
                raise FileNotFoundError(f"Data file not found at {data_path}")
            data = pd.read_csv(data_path)

            if data.empty:
                logging.error("The dataset is empty.")
                raise ValueError("Dataset is empty.")
            logging.info(f"Dataset loaded successfully with shape: {data.shape}")

            # Clean column names
            logging.info("Cleaning column names by stripping whitespaces.")
            data.columns = data.columns.str.strip()
            logging.debug(f"Column names after cleaning: {data.columns.tolist()}")

            # Save raw data
            os.makedirs(os.path.dirname(self.DataIngistion_config.rawdata_path), exist_ok=True)
            data.to_csv(self.DataIngistion_config.rawdata_path, index=False)
            logging.info("Successfully saved raw data to Artifacts folder.")

            # Perform train-test split
            logging.info("Performing train-test split.")
            if len(data) < 2:
                logging.error("Insufficient data for train-test split.")
                raise ValueError("Dataset has fewer than two rows.")

            train_data, test_data = train_test_split(data, test_size=0.25, random_state=42)
            logging.info("Train-test split completed successfully.")

            # Save train and test data
            train_data.to_csv(self.DataIngistion_config.traindata_path, index=False)
            test_data.to_csv(self.DataIngistion_config.testdata_path, index=False)
            logging.info("Successfully saved train and test data to Artifacts folder.")

            logging.info("Data ingestion completed successfully.")
            return train_data, test_data  # Return for further processing

        except Exception as e:
            logging.error("Exception occurred during the data ingestion stage.")
            raise customexception(e, sys)
