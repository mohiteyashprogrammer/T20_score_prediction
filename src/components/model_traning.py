import os
import sys
import pandas as pd
import numpy as numpy
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException

from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBRFRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV


from src.utils import model_traning,load_object,save_object

@dataclass
class ModelTraningConfig:
    traning_model_file_path = os.path.join("artifcats","model.pkl")


class ModelTraning:

    def __init__(self):
        self.model_traning_config = ModelTraningConfig()

    def initated_model_traning(self,train_array,test_array):

        '''
        This Function Will TRain Multiple Model And Give Us Best

        '''
        try:
            logging.info("Split Dependent and indipendent features")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "KNeighborsRegressor":KNeighborsRegressor(n_neighbors=10),
                "GradientBoostingRegressor":GradientBoostingRegressor(
                    n_estimators= 1000,
                    max_depth=12,
                    learning_rate=0.2,
                    random_state=1  
                    )
            }

            model_report:dict = model_traning(X_train, y_train, X_test, y_test, models)
            print(model_report)
            print("="*100)
            logging.info(f"Model Report: {model_report}")

            # To Get The Best Model
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f"Best Model Found, Model Name : {best_model_name}, R2 Score : {best_model_score}")
            print("\n***************************************************************\n")
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')


            save_object(filepath = self.model_traning_config.traning_model_file_path,
            obj = best_model)


        except Exception as e:
            logging.info("Exception Occured at Model Traning")
            raise CustomException(e,sys)





 