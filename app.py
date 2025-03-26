from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load trained model and imputer (for handling missing values)
model_path = "myocardial_model.pkl"
imputer_path = "imputer.pkl"

if os.path.exists(model_path) and os.path.exists(imputer_path):
    model = joblib.load(model_path)
    imputer = joblib.load(imputer_path)
else:
    model = None
    imputer = None

@app.route('/')
def home():
    # Define the top 15 features and their valid ranges
    features_info = [
        ("AGE", "18 - 100"),
        ("SEX", "0 = Female, 1 = Male"),
        ("SIM_GIPERT", "0 = No, 1 = Yes"),
        ("STENOK_AN", "0 = No, 1 = Yes"),
        ("FK_STENOK", "0 = No, 1 = Yes"),
        ("IBS_POST", "0 = No, 1 = Yes"),
        ("IBS_NASL", "0 = No, 1 = Yes"),
        ("K_BLOOD", "3.5 - 5.5 (mmol/L)"),
        ("L_BLOOD", "3.0 - 10.0 (10^9/L)"),
        ("ROE", "1 - 30 (mm/hr)"),
        ("S_AD_KBRIG", "90 - 180 (mmHg)"),
        ("D_AD_KBRIG", "60 - 120 (mmHg)"),
        ("GIPO_K", "0 = No, 1 = Yes"),
        ("GIPER_NA", "0 = No, 1 = Yes"),
        ("CHOL", "3.0 - 8.0 (mmol/L)")
    ]

    return render_template("home.html", features_info=features_info)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not model or not imputer:
            return jsonify({"error": "Model or imputer file not found!"})

        # Debugging: Print received form data
        print("Received Form Data:", request.form)

        # Extract input features from form
        features = [
            float(request.form.get("AGE", 0)),  
            float(request.form.get("SEX", 0)),  
            float(request.form.get("SIM_GIPERT", 0)),  
            float(request.form.get("STENOK_AN", 0)),  
            float(request.form.get("FK_STENOK", 0)),  
            float(request.form.get("IBS_POST", 0)),  
            float(request.form.get("IBS_NASL", 0)),  
            float(request.form.get("K_BLOOD", 0)),  
            float(request.form.get("L_BLOOD", 0)),  
            float(request.form.get("ROE", 0)),  
            float(request.form.get("S_AD_KBRIG", 0)),  
            float(request.form.get("D_AD_KBRIG", 0)),  
            float(request.form.get("GIPO_K", 0)),  
            float(request.form.get("GIPER_NA", 0)),  
            float(request.form.get("CHOL", 0))
        ]

        # Convert to NumPy array and preprocess
        input_data = np.array(features).reshape(1, -1)
        input_data = imputer.transform(input_data)  # Handle missing values

        # Make prediction
        prediction = model.predict(input_data)[0]
        result = "High Risk of Myocardial Infarction" if prediction == 1 else "Low Risk of Myocardial Infarction"

        return render_template("result.html", prediction=result)

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
