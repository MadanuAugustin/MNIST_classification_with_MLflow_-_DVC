
from src.MNIST import logger
from ExceptionFile.exception import CustomException
import sys
from src.MNIST.pipeline.stage_01_data_ingestion import DataIngestionPipeline

STAGE_NAME = 'DataIngestioinPipeline'


try:
    logger.info(f'-------------{STAGE_NAME} started---------------------!')
    data_ingestion = DataIngestionPipeline()
    data_ingestion.main()
    logger.info(f'-------------------{STAGE_NAME} completed---------------!')
except Exception as e:
    raise CustomException(e, sys)