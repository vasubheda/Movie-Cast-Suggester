import sys
import os
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils.common import save_object

from sklearn.neighbors import NearestNeighbors
import numpy as np

@dataclass
class ModelTrainerConfig:
    trained_model_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    def initiate_model_trainer(self,df_array):
        try:
            logging.info("Model training started......")
            
            knn_model = NearestNeighbors(n_neighbors=1,metric='cosine')
            knn_model.fit(df_array)
            
            logging.info(".........Model Training is Completed")
                        
            save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=knn_model
            )
            logging.info("Model Saved.")
            
        except Exception as e:
            raise CustomException(e,sys)