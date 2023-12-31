from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split, cross_val_score
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.tree import DecisionTreeRegressor
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        location = request.form['locality']
        sqft = float(request.form['sqft'])
        bath = int(request.form['bath'])
        bhk = int(request.form['bhk'])
        prediction = predict_price(location, sqft, bath, bhk)
        return render_template('predict.html', prediction=prediction)
    return render_template('predict.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')

import joblib

def predict_price(location, sqft, bath, bhk):
    X = joblib.load('Eng_X.joblib')
    lr_clf = joblib.load('linear_regression_model.joblib')
    loc_index = np.where(X.columns==location)[0][0]
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    return (lr_clf.predict([x])[0] * 1.25)

if __name__ == '__main__':
    app.run(debug=True)
