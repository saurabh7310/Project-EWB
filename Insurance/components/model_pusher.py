import os,sys 
import numpy as np
import pandas as pd
from Insurance import utils
from typing import Optional
from Insurance.logger import logging
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from Insurance.utils import load_object, save_object
from Insurance.config import TARGET_COLUMN
from Insurance.predictor import ModelResolver
from Insurance.exception import InsuranceException
from sklearn.linear_model import LinearRegression
from Insurance.entity import artifact_entity,config_entity
from Insurance.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ModelPusherArtifact
from Insurance.entity.config_entity import ModelPusherConfig
from Insurance.predictor import ModelResolver

class ModelPusher:
    def __init__(self, 
                 model_pusher_config: ModelPusherConfig,
                 data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_pusher_config = model_pusher_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = ModelResolver(model_registry= self.model_pusher_config.saved_model_dir)
        except Exception as e:
            raise InsuranceException(e, sys)
        

    def intitate_model_pusher(self)->ModelPusherArtifact:
        try:
            # Load Model and Target Encoder data
            transformer = load_object(file_path= self.data_transformation_artifact.transform_object_path)
            model = load_object(file_path= self.model_trainer_artifact.model_path)
            target_encoder = load_object(file_path= self.data_transformation_artifact.target_encoder_path)

            # Model Pusher Directory
            save_object(file_path= self.model_pusher_config.pusher_transformer_path, obj= transformer)
            save_object(file_path= self.model_pusher_config.pusher_model_path, obj= model)
            save_object(file_path= self.model_pusher_config.pusher_target_encoder_path, obj= target_encoder)

            # Save Model
            transform_path = self.model_resolver.get_latest_save_transformer_path()
            model_path = self.model_resolver.get_latest_save_model_path()
            target_encoder_path = self.model_resolver.get_latest_save_target_encoder_path()

            save_object(file_path= transform_path, obj= transformer)
            save_object(file_path= model_path, obj= model)
            save_object(file_path= target_encoder_path, obj= target_encoder)    

            model_pusher_artifact = ModelPusherArtifact(pusher_model_dir= self.model_pusher_config.pusher_model_dir,
                                                        saved_model_dir= self.model_pusher_config.saved_model_dir)
            
            return model_pusher_artifact    

        except Exception as e:
            raise InsuranceException(e, sys)