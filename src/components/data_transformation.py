import os
import sys
from dataclasses import dataclass
import ast
from sklearn.preprocessing import LabelEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

import pandas as pd
import numpy as np

from src.logger import logging
from src.exception import CustomException
from src.utils.common import save_object

nltk.download('punkt')
nltk.download('stopwords')

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        words = word_tokenize(text)
        words = [word for word in words if word not in stopwords.words('english')]
        text = ' '.join(words)
        return text
        
    def get_data_transformer_object(self):
        '''
        Responsible for data transformation
        '''
        try:
            numerical_features=['budget','popularity','runtime','vote_average']
            text_feature='Story'
                
            num_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
                ]
            )
            text_pipeline=Pipeline(
                steps=[
                ("preprocessor",FunctionTransformer(lambda x: x.apply(self.preprocess_text),validate=False)),
                ("vectorizer",TfidfVectorizer())
                ]
            )
            
            logging.info(f"Numerical Columns:{numerical_features}")
            logging.info(f"Story Column:{text_feature}")
            
            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_features),
                ("text_pipeline",text_pipeline,text_feature)
                ]
            )
            
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,data_path):
        try:
            df=pd.read_csv(data_path)
            
            logging.info("Reading the dataset completed")
            logging.info("obtaining preprocessing object")
            
            preprocessing_obj=self.get_data_transformer_object()
            
            # additional_features=df[['budget','popularity','runtime','vote_average','year']]
            
            logging.info(
                f"Applying preprocessing object"
            )
            
            df_to_arr= preprocessing_obj.fit_transform(df)
            
            logging.info(f"Saved preprocessing object.")
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            return(
                df_to_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e,sys)
        
        
            
            