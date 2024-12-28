import pandas as pd
import numpy as np
import sys
import os
from Concrete_Strenght_prediction.utils.utils import save_object
from sklearn.ensemble import RandomForestRegressor
from Concrete_Strenght_prediction.logger import logging
from Concrete_Strenght_prediction.exception import customexception  
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error   


class ModelTrainerConfig:
    """
    Configuration class to store paths for saving models.
    """
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    """
    Class for model training, evaluation, and saving the trained model.
    """
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info('Splitting dependent and independent variables from the transformed data')
            
            # Split the train_arr into features (X) and target variable (y)
            X_train = train_arr[:, :-1]  # All columns except the last one (features)
            y_train = train_arr[:, -1]   # Last column as target (target variable)

            # Split the test_arr into features (X) and target variable (y)
            X_test = test_arr[:, :-1]    # All columns except the last one (features)
            y_test = test_arr[:, -1]     # Last column as target (target variable)
            
            # Initialize the model with hyperparameters
            model = RandomForestRegressor(
                bootstrap=False, 
                max_depth=30, 
                max_features='log2', 
                min_samples_leaf=1, 
                min_samples_split=2, 
                n_estimators=300
            )
            
            # Fit the model to the training data
            model.fit(X_train, y_train)
            
            # Predict on the test data
            y_pred = model.predict(X_test)
            
            # Calculate performance metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2_value = r2_score(y_test, y_pred)
            
            # Log the model performance
            logging.info(f'Performance: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2_value * 100:.2f}%')
            
            # Save the trained model to a file
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=model)
            
            logging.info(f"Model training completed successfully. Model performance: R²={r2_value * 100:.2f}%")

        except Exception as e:
            logging.error('Exception occurred during Model Training')
            raise customexception(e, sys)
