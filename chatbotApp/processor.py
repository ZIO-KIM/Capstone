import flask
import pickle
import pandas as pd
from pycaret.classification import *
from pororo import Pororo
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
# define sts
sts = Pororo(task="semantic_textual_similarity", lang="ko")


# input - 1. breed
def check_breed(breed):   

  if breed in breed_list: 
    response = "반려견의 종이 {}로 입력되었습니다. 다음으로 나이를 입력하시려면 '나이 입력'을 입력해 주세요.".format(breed)

  elif breed == '종 모름':
    response = "'믹스'로 계속하시겠습니까? 계속하실 경우 1, 종료하실 경우 2를 입력해주세요."
  
  elif breed == '1':
      breed = 'mixed'
      response = "반려견의 종이 {}로 입력되었습니다. 다음으로 나이를 입력하시려면 '나이 입력'을 입력해 주세요.".format(breed)
  
  elif breed == '2':
      response = "의사소통 서비스를 종료합니다. 만족할 만한 서비스를 제공해 드리지 못해 죄송합니다."

  else: 
    response = "반려견의 종이 의사소통이 가지고 있는 데이터베이스에 들어 있지 않습니다. 죄송합니다.\n\n'믹스'로 계속하시겠습니까? 계속하실 경우 1, 종료하실 경우 2를 입력해주세요."

  return response, breed

# input - 2. age
def check_age(age): 

  if age == '나이 모름': 
    response = "반려견의 나이대는 어느 정도인가요? 새끼일 경우 '새끼', 중반 정도일 경우 '중반', 노령일 경우 '노령'을 입력해주세요. 나이대도 잘 모를 경우, '나이대 모름' 을 입력해주세요."

  elif age == '새끼': 
    age = 0
    response = "반려견의 나이가 1세 미만으로 입력되었습니다. 다음으로 성별을 입력하시려면 '성별 입력'을 입력해 주세요."

  elif age == '중반': 
    age = 3
    response = "반려견의 나이가 {}세로 입력되었습니다. 다음으로 성별을 입력하시려면 '성별 입력'을 입력해 주세요.".format(age)
  
  elif age == '노령': 
    age = 7
    response = "반려견의 나이가 {}세로 입력되었습니다. 다음으로 성별을 입력하시려면 '성별 입력'을 입력해 주세요.".format(age)

  elif age == '나이대 모름': 
    response = "나이를 생략하고 계속하시겠습니까? 계속하실 경우 '계속', 종료하실 경우 '종료'를 입력해주세요."
  
  elif age == '계속': 
    age = 'unknown'
    response = "반려견의 나이가 {}으로 입력되었습니다. 다음으로 성별을 입력하시려면 '성별 입력'을 입력해 주세요.".format(age)

  elif age == '종료': 
    response = "의사소통 서비스를 종료합니다. 만족할 만한 서비스를 제공해 드리지 못해 죄송합니다."

  else: 
    print("반려견의 나이가 {}세로 입력되었습니다. 다음으로 성별을 입력하시려면 '성별 입력'을 입력해 주세요.".format(age))

  return response, age

# input - 3. sex
def check_sex(sex):
  print("반려견의 성별은 무엇인가요? (수컷일 경우 M, 암컷일 경우 F, 중성화 되었을 경우 1을 입력해주세요.)")
  print("성별을 모를 경우, '모름'을 입력해주세요.")
  
  if sex == 'M': 
    print("반려견의 성별이 {}으로 입력되었습니다.".format(sex))

  elif sex == 'F':
    print("반려견의 성별이 {}으로 입력되었습니다.".format(sex))

  elif sex == '1':
    print("반려견의 성별이 {}으로 입력되었습니다.".format(sex))

  elif sex == '모름': 
    print("성별을 생략하고 계속하시겠습니까? 계속하실 경우 1, 종료하실 경우 2를 입력해주세요.")
    answer = input()
    if answer == '1':
      sex = 'unknown'
      print("반려견의 성별이 {}으로 입력되었습니다.".format(sex))
    else: 
      print("의사소통 서비스를 종료합니다. 만족할 만한 서비스를 제공해 드리지 못해 죄송합니다.")

  return sex

# input - 4. symptoms
def check_symptoms(symptoms):
  print("반려견이 보이는 증상을 단어 혹은 문장으로 입력해주세요.")

  if symptoms in sample_symptoms_dict:  # 증상 dictionary key에 입력된 증상이 있을 경우
    print("의사소통 데이터베이스의 다음과 같은 증상이 입력하신 증상과 유사한 증상으로 확인되었습니다.")
    sim_symp = sample_symptoms_dict[symptoms] # 해당 key 에 매핑된 유사증상 list 불러오기
    for (i, item) in enumerate(sim_symp, start=1):
        print(i, ": {}".format(item))
    print("원하시는 증상의 번호를 입력해 주세요. 만약 일치하는 증상이 없어 종료를 원하시면 -1을 입력해 주세요.")
    answer = input()
    if answer != '-1': 
      symptoms = sim_symp[int(answer) - 1] # 해당 인덱스의 증상 뽑아오기
      print("반려견의 증상이 {} (으)로 입력되었습니다.".format(symptoms))
    else: 
      print("의사소통 데이터베이스에 입력하신 증상이 없습니다. 불편을 드려 죄송합니다.")
      print("의사소통 서비스를 종료합니다. 만족할 만한 서비스를 제공해 드리지 못해 죄송합니다.")

  elif symptoms in symp_list:  # DB에 가지고 있는 증상과 바로 일치할 경우
    print("반려견의 증상이 {} (으)로 입력되었습니다.".format(symptoms))

  else: # 없는 경우 - pororo 호출
    print("의사소통 데이터베이스에 있는 증상과 유사도 검사를 실시합니다. 조금 시간이 걸릴 수 있습니다.")

    sim_symp = []
    for elem in symp_list: 
      if sts(symptoms,elem) >= 0.5: # 유사도 0.5 이상이면 append
        sim_symp.append(elem)

    if sim_symp: # if sim_symp list not empty
      print("기다려 주셔서 감사합니다. 의사소통 데이터베이스의 다음과 같은 증상이 입력하신 증상과 유사한 증상으로 확인되었습니다.")
      for (i, item) in enumerate(sim_symp, start=1):
        print(i, ": {}".format(item))
      print("원하시는 증상의 번호를 입력해 주세요. 만약 일치하는 증상이 없어 종료를 원하시면 -1을 입력해 주세요.")
      answer = input()
      if answer != -1: 
        symptoms = sim_symp[int(answer) - 1] # 해당 인덱스의 증상 뽑아오기
        print("반려견의 증상이 {} (으)로 입력되었습니다.".format(symptoms))
      else: 
        print("의사소통 서비스를 종료합니다. 만족할 만한 서비스를 제공해 드리지 못해 죄송합니다.")

    else:  # 유사한 증상으로 뽑힌 단어가 없으면 (list empty)
      print("의사소통 데이터베이스에 입력하신 증상이 없습니다. 불편을 드려 죄송합니다.")
      print("빠른 시일 내 가까운 동물병원 방문을 추천드립니다.")
      print("의사소통 서비스를 종료합니다. 만족할 만한 서비스를 제공해 드리지 못해 죄송합니다.")

  return symptoms

def predict(breed, age, sex, symptoms): 
  breed_checked = check_breed(breed)
  age_checked = check_age(age)
  sex_checked = check_sex(sex)
  symptom_checked = check_symptoms(symptoms)

  X =  {'Breed' : [breed_checked], 'Age'	: [int(age_checked)], 'Sex' : [sex_checked], 'Symptoms' : [symptom_checked]  }
  X = pd.DataFrame(X)

  y_pred = predict_model(model, X)
  prediction = y_pred['Label'][0]

  return prediction

# def chatbot_response():  # 추후 수정 - 마지막 세부질병 출력까지 파이프라인에 추가되면 add