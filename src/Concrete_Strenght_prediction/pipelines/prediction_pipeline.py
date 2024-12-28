import numpy as np
import pandas as pd
import sys
import os
from Concrete_Strenght_prediction.logger import logging
from Concrete_Strenght_prediction.exception import customexception
from Concrete_Strenght_prediction.utils.utils import load_object

class Prediction_pipeline:
    def __init__(self, model_path="Artifacts/model.pkl", preprocessor_path="Artifacts/preprocessor.pkl"):
        try:
            self.model_path = model_path
            self.preprocessor_path = preprocessor_path
            
            logging.info(f"Loading model from: {self.model_path}")
            self.model = load_object(self.model_path)
            
            logging.info(f"Loading preprocessor from: {self.preprocessor_path}")
            self.preprocessor = load_object(self.preprocessor_path)
        except Exception as e:
            logging.error("Error initializing Prediction_pipeline")
            raise customexception(e, sys)
        
    def preprocess_data(self, input_data):
        """
        Preprocesses the input data using the loaded preprocessor.
        """
        try:
            logging.info("Preprocessing the data")
            logging.debug(f"Input data before preprocessing:\n{input_data}")
            
            processed_data = self.preprocessor.transform(input_data)
            
            logging.debug(f"Processed data after preprocessing:\n{processed_data}")
            return processed_data
        except Exception as e:
            logging.error("Error in preprocessing data")
            logging.error(f"Exception details: {e}")
            raise customexception(e, sys)
        
    def make_prediction(self, input_data):
        """
        Makes predictions on the input data.
        """
        try:
            processed_data = self.preprocess_data(input_data)
            predictions = self.model.predict(processed_data)
            logging.info("Prediction successful")
            return predictions
        except Exception as e:
            logging.error("Error in making predictions")
            logging.error(f"Exception details: {e}")
            raise customexception(e, sys)

class Custom_data:
    def __init__(self, cement, blast_furnace_slag, fly_ash, water, superplasticizer, 
                 coarse_aggregate, fine_aggregate, age):
        self.cement = cement
        self.blast_furnace_slag = blast_furnace_slag
        self.fly_ash = fly_ash
        self.water = water
        self.superplasticizer = superplasticizer
        self.coarse_aggregate = coarse_aggregate
        self.fine_aggregate = fine_aggregate
        self.age = age
        
    def validate_data(self):
        """
        Validates the input data for correctness.
        """
        if not all(isinstance(value, (int, float)) for value in [
            self.cement, self.blast_furnace_slag, self.fly_ash, self.water,
            self.superplasticizer, self.coarse_aggregate, self.fine_aggregate, self.age
        ]):
            raise ValueError("All input values must be of type int or float")
        
    def get_data_as_df(self):
        """
        Converts the input data to a pandas DataFrame.
        """
        try:
            self.validate_data()
            data_dict = {
                "cement": [self.cement],
                "blast_furnace_slag": [self.blast_furnace_slag],
                "fly_ash": [self.fly_ash],
                "water": [self.water],
                "superplasticizer": [self.superplasticizer],
                "coarse_aggregate": [self.coarse_aggregate],
                "fine_aggregate": [self.fine_aggregate  ],  # Ensure the column name is correct
                "age": [self.age],
            }
            df = pd.DataFrame(data_dict)
            
            # Strip leading/trailing spaces from the column names
            df.columns = df.columns.str.strip()  
            logging.debug(f"DataFrame after column name cleaning:\n{df}")
            
            return df
        except Exception as e:
            logging.error("Error converting data to DataFrame")
            logging.error(f"Exception details: {e}")
            raise customexception(e, sys)
