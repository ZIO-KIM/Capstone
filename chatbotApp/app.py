import flask
import pickle
import pandas as pd
from pycaret.classification import *

pipeline_path = 'model/model_pipeline'
model = load_model(pipeline_path)

app = flask.Flask(__name__, template_folder='templates')
@app.route('/', methods=['GET', 'POST'])

def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if flask.request.method == 'POST':
        breed = flask.request.form['breed']
        sex = flask.request.form['sex']
        age = flask.request.form['age']
        symptoms = flask.request.form['symptoms']
        X =  {'Breed' : [breed], 'Age' : [int(age)], 'Sex' : [sex], 'Symptoms' : [symptoms]}
        X = pd.DataFrame(X).reset_index(drop = True)
        y_pred = predict_model(model, X)
        prediction = y_pred['Label'][0]
        return flask.render_template('main.html',
                                     original_input={'breed':breed,
                                                     'sex':sex,
                                                     'age':age, 
                                                     'symptoms':symptoms},
                                     result=prediction,
                                     )

if __name__ == '__main__':
    app.run()