from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
import os

app = Flask(__name__)
data = pd.read_csv('Bangalore_Property_Data.csv')
pipe = pickle.load(open("LR_Bangalore_Model.pkl", "rb"))

@app.route('/')
def index():

    locations = sorted(data['location'].unique())
    return render_template('index.html', locations = locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath =  request.form.get('bath')
    sqft = request.form.get('total_sqft')

    try:
        bhk = float(bhk)
        bath = int(bath)
        sqft = float(sqft)
    except ValueError:
        return "Please enter valid numeric value for BHK, Bathrooms and Sq.ft"

    print(location, bhk, bath, sqft)
    input = pd.DataFrame([[location,sqft,bath, bhk]],columns = ['location', 'total_sqft','bath','BHK'])
    prediction = pipe.predict(input)[0]

    return str(np.round(prediction,2))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)