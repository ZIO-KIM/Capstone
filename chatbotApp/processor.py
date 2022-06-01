import flask
import pickle
import pandas as pd
from pycaret.classification import *
import json

pipeline_path = 'model/model_pipeline_remove4'
model = load_model(pipeline_path)


# final data import
data = pd.read_csv('data/final_data.csv', encoding = 'cp949')
data.drop(columns=['Unnamed: 0'], axis = 1, inplace = True)
# breed list
breed_list = data['Breed'].unique().tolist()
# symp list
symp_list = data['Symptoms'].unique().tolist()
# 증상 dictionary json load
with open('data/지식인_증상_유사단어_dictionary_Pororo.json','r') as f:
    sample_symptoms_dict = json.load(f)


def predict(conv_data): 
  X =  {'Breed' : conv_data['breed'], 'Age'	: conv_data['age'], 'Sex' : conv_data['sex'], 'Symptoms' : conv_data['symptoms'] }
  X = pd.DataFrame([X])

  y_pred = predict_model(model, X)
  prediction = y_pred['Label'][0]

  return prediction

# def chatbot_response():  # 추후 수정 - 마지막 세부질병 출력까지 파이프라인에 추가되면 add