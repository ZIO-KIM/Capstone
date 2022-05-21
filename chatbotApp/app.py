import flask
import pickle
import pandas as pd
from pycaret.classification import *
from pororo import Pororo
import json

# processor
import processor

pipeline_path = 'model/model_pipeline_remove4'
model = load_model(pipeline_path)


# final data import
data = pd.read_csv('data/final_data.csv', encoding = 'cp949')
data.drop(columns=['Unnamed: 0'], axis = 1, inplace = True)
breed_list = data['Breed'].unique().tolist()
# 증상 dictionary json load
with open('data/지식인_증상_유사단어_dictionary_Pororo.json','r') as f:
    sample_symptoms_dict = json.load(f)

#Flask initialization
app = flask.Flask(__name__, template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
# def main():
#     if flask.request.method == 'GET':
#         return(flask.render_template('index_test.html')) # main.html & index_test.html 혼용
#     if flask.request.method == 'POST':
#         # 종 입력
#         breed = flask.request.form['breed']
#         sex = flask.request.form['sex']
#         age = flask.request.form['age']
#         symptoms = flask.request.form['symptoms']

#         prediction = processor.predict(breed, age, sex, symptoms)


#         return flask.render_template('index_test.html',  # main.html & index_test.html 혼용
#                                      original_input={'breed':breed,
#                                                      'sex':sex,
#                                                      'age':age, 
#                                                      'symptoms':symptoms},
#                                      result=prediction,
#                                      )

# @app.route("/")
# def index():
# 	flask.render_template("index_test.html")

# @app.route("/get", methods=['GET', 'POST'])
# def chatbot_response(): 
#      msg = flask.request.form["msg"]
#      response = "종이 {}로 입력되었습니다".format(msg)
#      return response

@app.route("/breed", methods=['GET', 'POST'])
def input_breed():
    if flask.request.method == 'GET':
        return(flask.render_template('index_test.html')) # main.html & index_test.html 혼용
    if flask.request.method == 'POST':
        breed = flask.request.form["breed"]
        
        if breed == 'start':
            response = "반려견의 종은 무엇인가요? (ex. 말티즈, 보더콜리 등) <br> 종을 모를 경우, '종 모름'을 입력해주세요"
        else: 
            response, breed = processor.check_breed(str(breed))

        return response

@app.route("/age", methods=['GET', 'POST'])
def input_age():
    age = flask.request.form["age"]
    if age == '나이 입력': 
        response = "반려견의 나이는 몇 살인가요? (숫자만 입력해주세요) <br> 나이를 모를 경우, '나이 모름'을 입력해주세요."
    else: 
        response, age = processor.check_age(int(age))

    return response

if __name__ == '__main__':
    app.run(debug=True)