from collections import namedtuple


DataIngestionConfig = namedtuple("DataIngestionConfig", [
    "raw_data_dir", "ingestion_train_dir", "ingestion_test_dir"])

DataValidationConfig = namedtuple("DataValidationConfig", ["schema_file_path","report_file_path","report_page_file_path"])

DataTransformationConfig = namedtuple("DataTransformationConfig", [
    "transformed_train_dir",
    "transformed_test_dir",
])

ModelTrainingConfig = namedtuple("ModelTrainingConfig", [
    "trained_model_file_path",
    "base_accuracy"
])

ModelTrainerConfig = namedtuple("ModelTrainerConfig", ["trained_model_file_path","base_accuracy"])

ModelEvaluationConfig = namedtuple("ModelEvaluationConfig", ["model_evaluation_file_path","time_stamp"])

TrainingPipelineConfig = namedtuple("TrainingPipelineConfig", [
    "artifact_dir"
])
