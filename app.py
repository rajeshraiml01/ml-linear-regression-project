from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application


ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()
        df = pd.DataFrame(data, index=[0])
        df = standard_scaler.transform(df)
        prediction = ridge_model.predict(df)
        return jsonify({'prediction': prediction.tolist()})
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)