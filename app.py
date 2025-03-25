#!/usr/bin/env python
# coding: utf-8

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__, template_folder='templates')

# Load trained model
model_path = "model.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None  # Handle missing model error

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not model:
            return jsonify({"error": "Model file not found!"})

        # Extract input features from form
        features = [
            float(request.form["AGE"]),
            float(request.form["SEX"]),
            float(request.form["SIM_GIPERT"]),
            float(request.form["S_DIA_B"]),
            float(request.form["CHOL"])
        ]
        
        # Convert to NumPy array
        input_data = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        return render_template("result.html", prediction=prediction)
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000, debug=True)
