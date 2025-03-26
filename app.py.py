#!/usr/bin/env python
# coding: utf-8

# In[3]:


from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

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
            float(request.form["S_AD_KBRIG"]),
            float(request.form["D_AD_KBRIG"]),
            float(request.form["K_BLOOD"]),
            float(request.form["ROE"]),
            float(request.form["L_BLOOD"]),
            float(request.form["CHOL"]),
            float(request.form["S_DIA_B"])
        ]
        
        # Convert to NumPy array
        input_data = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Convert numeric output to readable format
        result = "Diabetic" if prediction == 1 else "Non-Diabetic"
        
        return render_template("result.html", prediction=result)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)

