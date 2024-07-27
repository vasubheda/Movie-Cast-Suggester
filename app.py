from flask import Flask,request,render_template
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

@app.route('/recommend',methods=["GET","POST"])
def recommend_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=request.form.get('story')
        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(data)
        return render_template('home.html',results=results[0])

if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)
    