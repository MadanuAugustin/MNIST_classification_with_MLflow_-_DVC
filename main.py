
from src.MNIST import logger
from ExceptionFile.exception import CustomException
import sys
from src.MNIST.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from src.MNIST.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
STAGE_NAME = 'DataIngestioinPipeline'


try:
    logger.info(f'-------------{STAGE_NAME} started---------------------!')
    data_ingestion = DataIngestionPipeline()
    data_ingestion.main()
    logger.info(f'-------------------{STAGE_NAME} completed---------------!')
except Exception as e:
    raise CustomException(e, sys)



STAGE_NAME = 'PrepareBaseModelStage'

try:
    logger.info(f'---------------{STAGE_NAME} started----------------')
    prepare_base_model = PrepareBaseModelTrainingPipeline()
    prepare_base_model.main()
    logger.info(f'----------------------{STAGE_NAME} completed-------------')
except Exception as e:
    raise CustomException(e, sys)