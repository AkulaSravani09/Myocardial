from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__, template_folder='templates')

# Load trained model
MODEL_PATH = "model.pkl"

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        model = None
        print("Warning: Model file not found! Please ensure 'model.pkl' is available.")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model is not loaded. Please upload 'model.pkl'."})

    try:
        # Define required input features
        required_fields = ["AGE", "SEX", "SIM_GIPERT", "GB", "K_BLOOD"]
        features = []
        
        # Extract and validate input data
        for field in required_fields:
            value = request.form.get(field)
            if value is None or value.strip() == "":
                return jsonify({"error": f"Missing input: {field}"})
            try:
                features.append(float(value))
            except ValueError:
                return jsonify({"error": f"Invalid value for {field}. Must be a number."})
        
        # Convert input to NumPy array and reshape for prediction
        input_data = np.array(features).reshape(1, -1)
        
        # Perform prediction
        prediction = model.predict(input_data)[0]
        
        # Interpret prediction result
        result = "Diabetic" if prediction == 1 else "Non-Diabetic"
        
        return render_template("result.html", prediction=result)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))  # Default port 5000 if not specified
    app.run(host="0.0.0.0", port=port, debug=True)
