import pandas as pd
import numpy as np
import os, sys
from Insurance.entity import config_entity
from Insurance.entity import artifact_entity
from Insurance.exception import InsuranceException
from Insurance import utils
from Insurance.logger import logging
from sklearn.model_selection import train_test_split

class DataIngestion: 
    
    def __init__(self, data_ingestion_config: config_entity.DataIngestionConfig):
        try: 
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise InsuranceException(e, sys)
        
    def initiate_data_ingestion(self)->artifact_entity.DataIngestionArtifact:
        try:
            logging.info(f"Export collection data as pandas dataframe")
            df:pd.DataFrame = utils.get_collection_as_dataframe(
                database_name = self.data_ingestion_config.database_name,
                collection_name = self.data_ingestion_config.collection_name)
            logging.info(f"Save data in future store")

            # Replace na with NAN
            df.replace(to_replace= "na", value= np.NAN, inplace= True)

            # Save data in future Store
            logging.info(f"Create feature store folder if not available")
            feature_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_dir, exist_ok= True)
            logging.info(f"Save df to feature store folder")

            # Save df to feature store folder
            df.to_csv(path_or_buf= self.data_ingestion_config.feature_store_file_path, index= True)

            logging.info(f"Splitting our data into train and test set")
            train_df, test_df = train_test_split(df, test_size= self.data_ingestion_config.test_size, random_state= 1)

            logging.info(f"Create dataset directory folder if not exists")
            dataset_dir= os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dataset_dir, exist_ok= True)

            logging.info(f"Save dataset to feature store folder")
            train_df.to_csv(path_or_buf= self.data_ingestion_config.train_file_path,
                            index= False, 
                            header= True)
            test_df.to_csv(path_or_buf= self.data_ingestion_config.test_file_path,
                           index= False, 
                           header= True)
            
            # Prepare artifact folder
            data_ingestion_artifact = artifact_entity.DataIngestionArtifact(
                feature_store_file_path= self.data_ingestion_config.feature_store_file_path,
                train_file_path = self.data_ingestion_config.train_file_path,
                test_file_path= self.data_ingestion_config.test_file_path)
            
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        
        except Exception as e:
            raise InsuranceException(error_message= e, error_detail= sys)