import flask
import pickle
import pandas as pd
from pycaret.classification import *
from pororo import Pororo
import json
from flask_pymongo import PyMongo
from dooly import Dooly
import sys

sys.setrecursionlimit(50000)

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
# define sts
sts = Pororo(task="semantic_textual_similarity", lang="ko")


#Flask initialization
app = flask.Flask(__name__, template_folder='templates')

# # MongoDB
# app.config["MONGO_URI"] = "mongodb://localhost:27017/myDatabase"
# mongo = PyMongo(app)


# conversation data dictionary
conv_data = {}

@app.route('/', methods=['GET', 'POST'])

@app.route("/answer", methods=['GET', 'POST'])
def input():
    if flask.request.method == 'GET':
        return(flask.render_template('index_test.html')) # main.html & index_test.html 혼용
    if flask.request.method == 'POST':
        answer = flask.request.form["answer"]
        
        if answer == 'start':
            # response = str(sts("구토", "배 통증"))
            response = "반려견의 종은 무엇인가요? 다음 중 해당하는 종을 입력해주세요. <br>  (말티즈, 푸들, 포메라니안, 시추, 비숑프리제, 요크셔 테리어, 치와와, 스피츠, 골든 리트리버, 닥스훈트, 진도견, 웰시코기, 시바견, 코커스패니얼, 믹스)"

        elif answer in breed_list: 
            conv_data.update({"breed": answer})
            response = "반려견의 나이는 몇 살인가요? (1세 미만일 경우 0 입력)"
        
        elif answer.isnumeric() == True: 
            conv_data.update({"age": int(answer)})
            response = "반려견의 성별은 무엇인가요? 수컷은 M, 암컷은 F, 중성화 되었을 경우 -1으로 입력해주세요."
        
        elif answer == 'M' or answer == 'F' or answer == '-1': 
            if answer == '-1': 
                conv_data.update({'sex': 1})
            else: 
                conv_data.update({'sex': answer})
            response = "반려견이 보이는 증상을 단어 혹은 문장으로 입력해주세요."
        
        else: 
            # response = str(sts(answer, "구토"))
            if answer in sample_symptoms_dict:  # 증상 dictionary key에 입력된 증상이 있을 경우
                sim_symp = sample_symptoms_dict[answer] # 해당 key 에 매핑된 유사증상 list 불러오기
                answer = sim_symp[0]  # 일단 유사증상 중 0번째로 임의 치환 (사용자 인풋 안받기 위해)

                conv_data.update({"symptoms": answer})
                predicted_disease = processor.predict(conv_data)
                output_specified = str(output.loc[output['ICD']==predicted_disease,'질병명'].dropna().unique().tolist())
                response = "예상되는 질병은 {} 입니다. <br>  이 카테고리에 해당되는 질병에는 {} 가 있습니다.".format(predicted_disease, output_specified)

            elif answer in symp_list:  # DB에 가지고 있는 증상과 바로 일치할 경우
                conv_data.update({"symptoms": answer})
                predicted_disease = processor.predict(conv_data)
                output_specified = str(output.loc[output['ICD']==predicted_disease,'질병명'].dropna().unique().tolist())
                response = "예상되는 질병은 {} 입니다. <br>  이 카테고리에 해당되는 질병에는 {} 가 있습니다.".format(predicted_disease, output_specified)
            
            else: # 없는 경우 - pororo 호출
                sim_symp = []
                for elem in symp_list: 
                    if sts(answer,elem) >= 0.5: # 유사도 0.5 이상이면 append
                        sim_symp.append(elem)

                if sim_symp: 
                    answer = sim_symp[0]  # 일단 유사증상 중 0번째로 임의 치환 (사용자 인풋 안받기 위해)

                    conv_data.update({"symptoms": answer})
                    predicted_disease = processor.predict(conv_data)
                    output_specified = str(output.loc[output['ICD']==predicted_disease,'질병명'].dropna().unique().tolist())
                    response = "예상되는 질병은 {} 입니다. <br>  이 카테고리에 해당되는 질병에는 {} 가 있습니다.".format(predicted_disease, output_specified)
                else: # 유사도 0.5 이상인 증상이 없을 때
                    response = "의사소통 데이터베이스에 입력하신 증상이 없습니다. 불편을 드려 죄송합니다. <br> 빠른 시일 내에 가까운 동물병원 방문을 추천드립니다."

        return response

if __name__ == '__main__':
    app.run(debug=True)