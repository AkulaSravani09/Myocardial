#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load trained model and imputer
model = joblib.load("myocardial_model.pkl")
imputer = joblib.load("imputer.pkl")

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input features from form
        features = [
            float(request.form["AGE"]),
            float(request.form["SEX"]),
            float(request.form["SIM_GIPERT"]),
            float(request.form["STENOK_AN"]),
            float(request.form["FK_STENOK"]),
            float(request.form["IBS_POST"]),
            float(request.form["IBS_NASL"]),
            float(request.form["K_BLOOD"]),
            float(request.form["L_BLOOD"]),
            float(request.form["ROE"]),
            float(request.form["S_AD_KBRIG"]),
            float(request.form["D_AD_KBRIG"]),
            float(request.form["GIPO_K"]),
            float(request.form["GIPER_NA"])
        ]
        
        # Convert to NumPy array and handle missing values
        input_data = np.array(features).reshape(1, -1)
        input_data = imputer.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Convert numeric output to readable format
        result = "High Risk of Myocardial Infarction" if prediction == 1 else "Low Risk of Myocardial Infarction"

        return render_template("result.html", prediction=result)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)

