import pickle 
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, app, jsonify

 

app = Flask(__name__)
## Loading our trained ML model and scaler file for scaling the user data before prediction
model = pickle.load(open("lr_model.pkl", "rb"))
scalar = pickle.load(open("scaling.pkl", 'rb'))

## Will create app.route for first root(local url to return to homepage)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods = ['post'])

def predict_api():    ## Creating predict_api for our app
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))
    new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
    output = model.predict(new_data)
    print(output[0])
    return jsonify(output[0])



## To create a html form
@app.route("/predict", methods = ['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape(1, -1))
    print(final_input)
    output = model.predict(final_input)[0]
    return render_template("home.html", prediction_text ="house price predictionis {}".format(output))

## To run this 

if __name__ == "__main__":
    app.run(debug=True)



