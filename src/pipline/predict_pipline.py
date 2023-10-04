import os
import sys
import pandas as pd
import numpy as numpy
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object


class PredictPipline:

    def __init__(self):
        pass


    def prediction(self,features):
        '''
        This function Will Predict The Output 
        Based On Input Features

        '''
        try:
            ## This line of path code work i both windos and linex
            preprocessor_path = os.path.join("artifcats","preprocessor.pkl")
            model_path = os.path.join("artifcats","model.pkl")

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)

            return pred

        except Exception as e:
            logging.info("Error Occured in Prediction pipline Stage")
            raise CustomException(e, sys)
            