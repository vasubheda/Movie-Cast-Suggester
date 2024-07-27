import sys
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from src.utils.common import load_object
from src.exception import CustomException
from sklearn.feature_extraction.text import TfidfVectorizer

class PredictPipeline:
    def __init__(self):
        self.df=pd.read_csv("artifacts/data.csv")
    
    def predict(self,data):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            
            vec_story=load_object(file_path=os.path.join("artifacts","vectorizer.pkl"))
            story_vec=vec_story.transform([data])
            distances,indices=model.kneighbors(story_vec)
            recoms=self.df.iloc[indices[0]]['cast'].values
            return recoms
        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,story:str):
        self.story=story
    
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict={
                "story":[self.story]
            }
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e,sys)