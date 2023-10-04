import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifcats","preprocessor.pkl")


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    def get_preprocessor_object(self):

        '''
        This Function Will Give Preprocessor Object To Transform Data

        '''

        try:
            logging.info("Start Data Transformation")

            catigorical_features = ['batting_team', 'bowling_team']

            numerical_features = ['current_score', 'balls_left', 'wickets_left', 'current_run_rate','last_five_overs']

            # Create numeric pipline
            num_pipline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )

            # Create cotigorical pipline
            cato_pipline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot",OneHotEncoder(sparse=False,handle_unknown="ignore",drop="first")),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            ## create preprocessor object
            preprocessor = ColumnTransformer([
                ("num_pipline",num_pipline,numerical_features),
                ("cato_pipline",cato_pipline,catigorical_features)
            ])

            return preprocessor

            logging.info("Pipline Complited")

        except Exception as e:
            logging.info("Error Occured In Data Transformation Stage")
            raise CustomException(e, sys)



    def initated_data_transformation(self,train_path,test_path):

        '''
        This Method apply Transformation Object On Data and Transform data
        
        '''

        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info("reading TRaning and Testing Data Complited")
            logging.info(f'Train Dataframe Head : \n{train_data.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_data.head().to_string()}')


            logging.info("Obtaning Preprocessor Object")

            preprocessor_obj = self.get_preprocessor_object()

            target_column_name = "runs_x"
            drop_columns = [target_column_name]

            ## Splate data in to indipendent and dependent features
            input_features_train_data = train_data.drop(drop_columns,axis=1)
            target_feature_train_data = train_data[target_column_name]

            ## Splate data in to indipendent and dependent features
            input_features_test_data = test_data.drop(drop_columns,axis=1)
            target_feature_test_data = test_data[target_column_name]


            # apply Preprocessor Object
            input_features_train_arr = preprocessor_obj.fit_transform(input_features_train_data)
            input_features_test_arr = preprocessor_obj.transform(input_features_test_data)

            logging.info("apply Preprocessor Object On Train and Test Data")

            # Convert In To array To fast traning
            train_array = np.c_[input_features_train_arr,np.array(target_feature_train_data)]
            test_array = np.c_[input_features_test_arr,np.array(target_feature_test_data)]

            # Savepreprocessor in Pickel Pile
            save_object(filepath = self.data_transformation_config.preprocessor_obj_file_path,
             obj = preprocessor_obj)

            logging.info("Preprocessor Object File is Saved")

            return(
                train_array,
                test_array,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info("Error Occured Data Transformation Stage")
            raise CustomException(e, sys)



