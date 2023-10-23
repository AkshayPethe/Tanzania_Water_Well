import os
import sys
import numpy as np
import pandas as pd
import pickle

from src.logger import logging
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_model(X_train,X_test,y_train,y_test,models):
    try:
        model_report = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5,scoring = 'accuracy')

            
            print(f"Evaluation Metrics of {title}\n")
            print(f"Accuracy Score: {acc:.2f}")
            print(f"Mean CV Score:{cv_scores.mean():.2f}")

            model_report[model_name] = accuracy

            return model_report
        
    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)
    
     
    


