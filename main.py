from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

obj=DataIngestion()
data_pathh=obj.initiate_data_ingestion()

data_transformation= DataTransformation()
X_story,_,_=data_transformation.initiate_data_transformation(data_pathh)

modeltrainer=ModelTrainer()
modeltrainer.initiate_model_trainer(X_story)

