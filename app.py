from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

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
    app.run(debug=True)
