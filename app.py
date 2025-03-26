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
    try:
        model = joblib.load(model_path)
    except Exception as e:
        model = None
        print(f"Error loading model: {e}")
else:
    model = None
    print("Warning: Model file not found!")

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model file not found! Please upload 'model.pkl'."})

        # Extract and validate input features
        required_fields = ["AGE", "SEX", "SIM_GIPERT", "S_DIA_B", "CHOL"]
        features = []

        for field in required_fields:
            value = request.form.get(field)
            if value is None or value.strip() == "":
                return jsonify({"error": f"Missing input: {field}"})
            try:
                features.append(float(value))
            except ValueError:
                return jsonify({"error": f"Invalid value for {field}. Must be a number."})

        # Convert to NumPy array
        input_data = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_data)[0]

        return render_template("result.html", prediction=prediction)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    port = int(os.getenv("PORT", 10000))  # Use PORT from environment variable if available
    app.run(host="0.0.0.0", port=port, debug=True)
