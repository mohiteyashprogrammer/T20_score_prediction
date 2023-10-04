import os
import sys
import pandas as pd
import numpy as numpy
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation


@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join("artifcats","train.csv")
    test_data_path:str = os.path.join("artifcats","test.csv")
    raw_data_path:str = os.path.join("artifcats","raw.csv")


class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()


    def start_data_ingestion(self):

        '''
        This Method Will Load Data In the Pipline

        '''
        logging.info("Data Ingestion Method Started")
        try:
            data = pd.read_csv(os.path.join("notebook/data","T20_matches_data_clean.csv"))
            logging.info("Data Reading as pandas data frame")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path,index=False)

            logging.info("Apply Train Test Split")
            train_set,test_set = train_test_split(data,test_size=0.20,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Data Ingestion Complited")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
 
        except Exception as e:
            logging.info("Error Occured in Data Ingestion Stage")
            raise CustomException(e, sys)



if __name__=="__main__":

    ingestion = DataIngestion()
    train_data_path,test_data_path = ingestion.start_data_ingestion()
    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initated_data_transformation(train_data_path, test_data_path)
