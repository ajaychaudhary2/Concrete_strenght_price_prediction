from Concrete_Strenght_prediction.components.data_ingestion import DataIngistion
from Concrete_Strenght_prediction.components.data_transformation import DataTransformation
from Concrete_Strenght_prediction.components.model_trainer import ModelTrainer
from Concrete_Strenght_prediction.logger import logging
from Concrete_Strenght_prediction.exception import customexception


if __name__=="__main__":
    
    try:
        obj = DataIngistion()
        train_df ,test_df = obj.initiate_data_ingestion()
        
        
        # Step 2: Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr = data_transformation.initiate_data_transformation(train_df, test_df)
        
        
         # Step 3: Model Training
        model_trainer = ModelTrainer()
        model_trainer.initiate_model_training(train_arr, test_arr)  # Ensure this method is correctly defined

        print("Training pipeline executed successfully.")
    
    
    except customexception as e:
        logging.error(f"Custom Exception: {str(e)}")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

    