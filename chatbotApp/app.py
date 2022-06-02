import flask
import pickle
import pandas as pd
from pycaret.classification import *
# from pororo import Pororo
import json
# from flask_pymongo import PyMongo
# from dooly import Dooly
import sys
from flask import jsonify

sys.setrecursionlimit(50000)

# error logging
from logging.config import dictConfig

dictConfig({
    'version': 1,
    'formatters': {
        'default': {
            'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
        }
    },
    'handlers': {
        'file': {
            'level': 'INFO', # INFO level 이상의 데이터를 로깅
            # 로깅 파일이 계속 쌓이지 않도록 특정 용량을 넘으면 가장 오래된 로그가 삭제됨.
            'class': 'logging.handlers.RotatingFileHandler', 
            'filename': 'test_error.log', # 저장 경로
            'maxBytes': 1024 * 1024 * 5,  # 5 MB
            'backupCount': 5,
            'formatter': 'default',
        },
    },
    'root': {
        'level': 'INFO',
        'handlers': ['file']
    }
})

# processor
import processor

pipeline_path = 'model/model_pipeline_remove4'
model = load_model(pipeline_path)


# final data import
data = pd.read_csv('data/final_data.csv', encoding = 'cp949')
data.drop(columns=['Unnamed: 0'], axis = 1, inplace = True)
# output data import
output = pd.read_csv('data/출력 종합본.csv', encoding = 'euc-kr')
# breed list
breed_list = data['Breed'].unique().tolist()
# symp list
symp_list = data['Symptoms'].unique().tolist()
# 증상 dictionary json load
with open('data/지식인_증상_유사단어_dictionary_Pororo.json','r') as f:
    sample_symptoms_dict = json.load(f)
# 필터링 데이터 load
filter = pd.read_csv('data/필터링1차.csv', encoding = 'cp949')
filter_list = filter['입력'].unique().tolist()

# define 유사도 검사 model
from sentence_transformers import SentenceTransformer, util
sts = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS') 
# # 영문 모델 시도
# sts = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

#Flask initialization
app = flask.Flask(__name__, template_folder='templates')

# # MongoDB
# app.config["MONGO_URI"] = "mongodb://localhost:27017/myDatabase"
# mongo = PyMongo(app)


# conversation data dictionary
conv_data = {}

@app.route('/', methods=['GET', 'POST'])

@app.route("/breed",methods=['GET','POST'])
def breed_input():
    if flask.request.method == 'GET':
        return(flask.render_template('index_test.html'))
    if flask.request.method == 'POST':
        answer = flask.request.get_json()
        breed = answer['answer']
        print(breed)

        if breed in breed_list:
            conv_data.update({"breed": breed})
            return breed

        elif breed == 'mixed': 
            conv_data.update({"breed": 'mixed'})
            return '믹스'

        else: # 잘못된 입력
            return "wrong"
            

@app.route("/age",methods=['GET','POST'])
def age_input():
    if flask.request.method == 'GET':
        return(flask.render_template('index_test.html'))
    if flask.request.method == 'POST':
        answer = flask.request.get_json()
        age = answer['answer']
        print(age)
    
    if age.isdigit():
        conv_data.update({"age": int(age)})
        return str(age)
    
    elif age =='young':
        conv_data.update({'age':0})
        return '0'
    
    elif age=='midage':
        conv_data.update({'age':3})
        return '3'
    
    elif age=='old':
        conv_data.update({'age':7})
        return '7'


@app.route("/sex",methods=['GET','POST'])
def sex_input():
    if flask.request.method == 'GET':
        return(flask.render_template('index_test.html'))
    if flask.request.method == 'POST':
        answer = flask.request.get_json()
        sex = answer['answer']
        print(sex)
    
    if sex == 'unknown':
        conv_data.update({'sex':'unknown'})
        return '생략'

    elif sex == 'M': 
        conv_data.update({'sex':'M'})
        return '수컷'
    
    elif sex == 'F': 
        conv_data.update({'sex':'F'})
        return '암컷'
    
    elif sex == 'N': 
        conv_data.update({'sex':1})
        return '중성화'
    

@app.route("/symptom", methods=['GET', 'POST'])
def symptom_input():
    if flask.request.method == 'GET':
        return(flask.render_template('index_test.html')) # main.html & index_test.html 혼용
    if flask.request.method == 'POST':
        answer = flask.request.get_json()
        symptom = answer['answer']
        print(symptom)

        if symptom in symp_list:  # DB에 가지고 있는 증상과 바로 일치할 경우 - 이 경우에만 바로 출력
            conv_data.update({"symptoms": symptom})
            predicted_disease = processor.predict(conv_data)
            output_specified = str(output.loc[output['ICD']==predicted_disease,'질병명'].dropna().unique().tolist())
            # if (predicted_disease == "질병 분류를 특정할 수 없는 경우")
            response = "검사 결과, 예상되는 질병은 {} 입니다.".format(predicted_disease)
            return jsonify({'index': 'true', 'response': response, 'ICD': predicted_disease, 'list': output_specified})
        
        elif symptom in filter_list: # 필터링 데이터에 있는 입력과 바로 일치할 경우
            tmp = filter.loc[filter['입력']==symptom,:]
            real_symptom = filter.loc[tmp.index[0],"증상"]
            return jsonify({'index': 'false', 'list': real_symptom})

        elif symptom in sample_symptoms_dict:  # 증상 dictionary key에 입력된 증상이 있을 경우
            sim_symp = sample_symptoms_dict[symptom] # 해당 key 에 매핑된 유사증상 list 불러오기
            return jsonify({'index': 'false', 'list': sim_symp})

            # conv_data.update({"symptoms": answer})
            # predicted_disease = processor.predict(conv_data)
            # output_specified = str(output.loc[output['ICD']==predicted_disease,'질병명'].dropna().unique().tolist())
            # response = "예상되는 질병은 {} 입니다. <br>  이 카테고리에 해당되는 질병에는 {} 가 있습니다.".format(predicted_disease, output_specified)

        else: # 없는 경우 - sbert 호출
            vector1 = sts.encode(symptom) # 입력받은 증상 encode
            sim_symp = []
            for elem in symp_list: 
                vector2 = sts.encode(elem)
                sim = util.cos_sim(vector1, vector2)
                sim = sim.item()
                if sim >= 0.5: # 유사도 0.5 이상이면 append
                    sim_symp.append(elem)

            if sim_symp: 
                return jsonify({'index': 'false', 'list': sim_symp})
                
            else: # 유사도 0.5 이상인 증상이 없을 때
                response = "의사소통 데이터베이스에 입력하신 증상이 없습니다. 불편을 드려 죄송합니다. <br> 빠른 시일 내에 가까운 동물병원 방문을 추천드립니다."

            # 영문 모델 load test
            # query_embedding = sts.encode('How big is London')
            # passage_embedding = sts.encode(['London has 9,787,426 inhabitants at the 2011 census',
            #                       'London is known for its finacial district'])
            # response = util.dot_score(query_embedding, passage_embedding)

            return response

@app.route("/ICD_specify", methods=['GET', 'POST'])
def ICD_specify_print():
    if flask.request.method == 'GET':
        return(flask.render_template('index_test.html')) # main.html & index_test.html 혼용
    if flask.request.method == 'POST':
        answer = flask.request.get_json()
        disease_name = answer['answer']
        print(disease_name)

        tmp = output.loc[output['질병명']==disease_name,:]
        describe_disease = output.loc[tmp.index[0],"설명"]

        return describe_disease

@app.route("/check_if_disease_isin", methods=['GET', 'POST'])
def check_if_disease_isin():
    if flask.request.method == 'GET':
        return(flask.render_template('index_test.html')) # main.html & index_test.html 혼용
    if flask.request.method == 'POST':
        answer = flask.request.get_json()
        disease_name = answer['answer']
        print(disease_name)

        disease_list = output['질병명'].unique().tolist()
        
        if disease_name in disease_list: 
            return "true"
        else: 
            return "false"

@app.route("/check_if_symptom_isin", methods=['GET', 'POST'])
def check_if_symptom_isin():
    if flask.request.method == 'GET':
        return(flask.render_template('index_test.html')) # main.html & index_test.html 혼용
    if flask.request.method == 'POST':
        answer = flask.request.get_json()
        symptom_name = answer['answer']
        print(symptom_name)
        
        if symptom_name in symp_list: 
            return "true"
        else: 
            return "false"
        
if __name__ == '__main__':
    app.run(debug=True)