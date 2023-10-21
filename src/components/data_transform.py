import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
import pickle
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn import metrics
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, LabelBinarizer
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn import set_config

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self, X_train, X_test, y_train):
        try:
            logging.info("Data Transformation Initiated")
            
            # Transformation function
            def transform(X, col):
                value_counts = X[col].value_counts(normalize=True, ascending=False)
                cumulative_sum = value_counts.cumsum()
                selected_values = cumulative_sum[cumulative_sum <= 0.75].index
                encoded = X[col].apply(lambda x: 'Major' if x in selected_values else 'Minor')
                return encoded
            
            X_train['funder'] = transform(X_train, 'funder')
            X_train['installer'] = transform(X_train, 'installer')
            X_test['funder'] = transform(X_test, 'funder')
            X_test['installer'] = transform(X_test, 'installer')

            num_index = [1, 3, 4, 7, 8]
            cat_index = [0, 2, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

            cat_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore'))
            ])

            num_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy="median")),
                ('scaler', RobustScaler(with_centering=False))
            ])

            preprocessor = ColumnTransformer([
                ('cat_transformer', cat_transformer, cat_index),
                ('num_transformer', num_transformer, num_index)
            ], remainder='passthrough')

            label_binarizer = LabelBinarizer()
            label_binarizer = LabelBinarizer()
            y_train_encoded = label_binarizer.fit_transform(y_train)
            y_test_encoded = label_binarizer.transform(y_test)
            y_train_label = np.argmax(y_train_encoded, axis=1)
            y_test_label = np.argmax(y_test_encoded, axis=1)

            return preprocessor,
            logging.info("PipeLine is Completed")

    
        except Exception as e:
            logging.error("Error in Data Transformation Object")
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self,train_path,test_path):
            
            try:
                 train_df = pd.read_csv(train_path)
                 test_df = pd.read_csv(test_path)
                 
            except Exception as e:
                 logging.info("Exception occured in Data Transformation Initiation")
                 raise CustomException(e,sys)
            


# Initialize and use DataTransformation class
data_transformer = DataTransformation()
data_transformer.get_data_transformation_object(X_train, X_test, y_train)

# The rest of your Python script
# ...



