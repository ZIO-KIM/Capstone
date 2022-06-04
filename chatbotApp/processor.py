import flask
import pickle
import pandas as pd
from pycaret.classification import *
import json

pipeline_path = 'model/model_pipeline_final'
model = load_model(pipeline_path)


def predict(conv_data): 
  X =  {'Breed' : conv_data['breed'], 'Age'	: conv_data['age'], 'Sex' : conv_data['sex'], 'Symptoms' : conv_data['symptoms'] }
  X = pd.DataFrame([X])

  y_pred = predict_model(model, X)
  prediction = y_pred['Label'][0]

  return prediction

# def chatbot_response():  # 추후 수정 - 마지막 세부질병 출력까지 파이프라인에 추가되면 add