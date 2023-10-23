import os
import sys
import pandas as pd
import numpy as np


from src.logger import logging
from src.exception import CustomException

from src.components.data_transform import DataTransformation
from src.components.model_training import ModelTrainer
from src.components.data_ingestion import DataIngestion


if __name__ == '__main__':
    obj = DataIngestion()
    obj.initiate_data_ingestion()  # Initiating data ingestion
    
    # Now initiate data transformation
    data_transformation = DataTransformation()
    train_path = obj.ingestion_config.train_data_path
    test_path = obj.ingestion_config.test_data_path
    data_transformation.initiate_data_transformation(train_path, test_path)

    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training(input_feature_train_arr,input_feature_test_arr,input_target_train_arr,input_target_test_arr)