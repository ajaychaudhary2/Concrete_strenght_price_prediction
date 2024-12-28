import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from Concrete_Strenght_prediction.logger import logging
from Concrete_Strenght_prediction.exception import customexception  
from Concrete_Strenght_prediction.utils.utils import save_object

class DataTransformationConfig:
    """
    Configuration class for the Data Transformation process.
    """
    preprocessor_obj_file_path = os.path.join("Artifacts", "preprocessor.pkl")

class DataTransformation:
    """
    DataTransformation class to handle preprocessing of train and test datasets.
    """
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation(self, feature_columns):
        """
        Create and return a preprocessor pipeline for dynamically detected columns.
        """
        try:
            logging.info("Data transformation started")
            
            # Pipeline for numerical features
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='mean')),
                ("scaler", StandardScaler())
            ])

            logging.info("Preprocessing pipeline created.")

            # Combine pipelines into a ColumnTransformer
            preprocessor = ColumnTransformer(transformers=[ 
                ("num_pipeline", num_pipeline, feature_columns)
            ])

            return preprocessor

        except Exception as e:
            logging.error("Exception occurred during data transformation")
            raise customexception(e, sys)

    def remove_outliers(self, df):
        """
        Remove outliers from the dataset using the IQR method.
        """
        try:
            logging.info("Outlier detection started")

            # Calculate Q1, Q3 and IQR
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1

            # Filter out the outliers
            df_clean = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

            logging.info(f"Outliers removed: {df.shape[0] - df_clean.shape[0]} rows removed.")
            return df_clean

        except Exception as e:
            logging.error("Exception occurred during outlier removal")
            raise customexception(e, sys)

    def initiate_data_transformation(self, train_df, test_df):
        """
        Transform the train and test datasets using the preprocessing pipeline and remove outliers.
        """
        try:
            # Define the target column
            target_column_name = 'concrete_compressive_strength'

            # Remove outliers from train and test data
            train_df_clean = self.remove_outliers(train_df)
            test_df_clean = self.remove_outliers(test_df)

            # Detect numerical columns dynamically (after outlier removal)
            feature_columns = train_df_clean.select_dtypes(include=[np.number]).columns.tolist()
            if target_column_name in feature_columns:
                feature_columns.remove(target_column_name)

            logging.info(f"Feature columns detected: {feature_columns}")

            # Split input features and target variable for training and testing
            input_feature_train_df = train_df_clean[feature_columns]
            input_feature_test_df = test_df_clean[feature_columns]
            target_feature_train_df = train_df_clean[target_column_name]
            target_feature_test_df = test_df_clean[target_column_name]

            # Log the shape of datasets
            logging.info(f"Input training data shape: {input_feature_train_df.shape}")
            logging.info(f"Input testing data shape: {input_feature_test_df.shape}")

            # Apply the transformation pipeline
            preprocessor = self.get_data_transformation(feature_columns)
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            logging.info("Applying preprocessing on train and test datasets")

            # Combine transformed features and target variable into final arrays
            train_arr = np.c_[input_feature_train_arr, target_feature_train_df.values]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_df.values]

            # Save the preprocessor object
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )

            logging.info(f"Preprocessor object saved at: {self.data_transformation_config.preprocessor_obj_file_path}")

            return train_arr, test_arr

        except Exception as e:
            logging.error("Error occurred during data transformation")
            raise customexception(e, sys)
