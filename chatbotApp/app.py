import flask
import pickle
import pandas as pd
from pycaret.classification import *

# Use pickle to load in the pre-trained model.
with open(f'model/model_pipeline.pkl', 'rb') as f:
    model = pickle.load(f)

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
        X = pd.DataFrame(X)
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