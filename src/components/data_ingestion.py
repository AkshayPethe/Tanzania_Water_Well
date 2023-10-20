import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException

# Initialize the Data Ingestion Configuration
@dataclass
class DataIngestionconfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')

# Create a class for data ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionconfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Method Starts")
        try:
            X = pd.read_csv('Data/4910797b-ee55-40a7-8668-10efd5c1b960.csv')
            y = pd.read_csv('Data/0bf8bc6e-30d0-4c50-956a-603fc693d966.csv')

            # Concatenate X and Y for one Master DataFrame
            df = pd.concat([X, y], axis=1)

            # Dropping 'id' Column
            df.drop('id', axis=1, inplace=True)

            # Dropping Duplicates
            df.drop_duplicates(keep='first', inplace=True)
            data = df.copy()

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            train_set, test_set = train_test_split(df, test_size=0.3, stratify=y, random_state=101)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of Data is Completed")

        except Exception as e:
            logging.error('Exception occurred at Data Ingestion')
            raise CustomException(e, sys)

if __name__ == '__main__':
    obj = DataIngestion()
    obj.initiate_data_ingestion()
            
        
        
        
         