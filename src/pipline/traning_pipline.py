import os
import sys
import pandas as pd
import numpy as numpy
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_traning import ModelTraning



if __name__=="__main__":
    ingestion = DataIngestion()
    train_data_path,test_data_path = ingestion.start_data_ingestion()
    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initated_data_transformation(train_data_path, test_data_path)
    model_traning = ModelTraning()
    model_traning.initated_model_traning(train_arr, test_arr)
