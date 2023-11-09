import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)
data = pd.read_csv('database/Advertising.csv')
model = joblib.load("model/SalesPrediction.pkl")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    TV = request.form.get('TV')
    Radio = request.form.get('Radio')
    Newspaper = request.form.get('Newspaper')

    input_data = pd.DataFrame([[TV, Radio, Newspaper]], columns=[
                              'TV', 'Radio', 'Newspaper'])

    prediction = model.predict(input_data)[0]
    prediction = round(prediction)
    
    st= str(prediction)
    st2=" Sales expected"
    print(st+st2)
    return (st+st2)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
