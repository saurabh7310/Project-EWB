from Insurance.logger import logging
from Insurance.exception import InsuranceException
from Insurance.utils import get_collection_as_dataframe
from Insurance.entity.config_entity import DataIngestionConfig
from Insurance.entity import config_entity
import os, sys
from Insurance.components.data_ingestion import DataIngestion
from Insurance.components.data_validation import DataValidation
from Insurance.components.data_transformation import DataTransformation
from Insurance.components.model_trainer import ModelTrainer
from Insurance.components.model_evaluation import ModelEvaluation

# def test_logger_and_exception():
#     try:
#         logging.info("Stating the test_logger_and_exception")
#         result = 3/0
#         print(result)
#         logging.info("Ending the test_logger_and_exception")
#     except Exception as e:
#         logging.debug(str(e))
#         raise InsuranceException(e, sys)


if __name__ == "__main__":
    try:
        # test_logger_and_exception()
        # get_collection_as_dataframe(database_name="INSURANCE", collection_name="INSURANCE_PROJECT")
        training_pipeline_config = config_entity.TrainingPipelineConfig()
        data_ingestion_config = config_entity.DataIngestionConfig(training_pipeline_config = training_pipeline_config)
        print(data_ingestion_config.to_dict())

        data_ingestion = DataIngestion(data_ingestion_config= data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

        # Data Validation
        data_validation_config= config_entity.DataValidationConfig(training_pipeline_config= training_pipeline_config)
        data_validation = DataValidation(data_validation_config= data_validation_config,
                                         data_ingestion_artifact= data_ingestion_artifact)
        
        data_validation_artifact = data_validation.initiate_data_validation()

        # Data Transformation 
        data_transformation_config= config_entity.DataTransformationConfig(training_pipeline_config= training_pipeline_config)
        data_transformation = DataTransformation(data_transformation_config= data_transformation_config,
                                                 data_ingestion_artifact= data_ingestion_artifact)
        
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        
        # Model trainer
        model_trainer_config = config_entity.ModelTrainingConfig(training_pipeline_config= training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config = model_trainer_config,
                                     data_transformation_artifact = data_transformation_artifact)
        
        model_trainer_artifact = model_trainer.initiate_model_trainer()

        # Model Evaluation
        model_eval_config = config_entity.ModelEvaluationConfig(training_pipeline_config= training_pipeline_config)
        model_eval = ModelEvaluation(model_eval_config = model_eval_config,
                                     data_ingestion_artifact = data_ingestion_artifact,
                                     data_transformation_artifact = data_transformation_artifact,
                                     model_trainer_artifact = model_trainer_artifact)

        model_evl_artifact = model_eval.intitate_model_evaluation()
    except Exception as e:
        print(e)

