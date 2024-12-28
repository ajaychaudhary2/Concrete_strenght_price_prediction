import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


from Concrete_Strenght_prediction.logger import logging
from Concrete_Strenght_prediction.exception import customexception

def save_object(file_path, obj):
    """
    Saves a Python object to a specified file path using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise customexception(e, sys)
    
    


def load_object(file_path):
    """
    Loads a Python object from a specified file path using pickle.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            logging.info(f"Loading object from: {file_path}")
            return pickle.load(file_obj)
    except Exception as e:
        logging.error(f"Exception occurred while loading object from {file_path}: {e}")
        raise customexception(e, sys)