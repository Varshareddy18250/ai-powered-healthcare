from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("disease_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symptoms = data.get("symptoms", [])
    
    if not symptoms:
        return jsonify({"error": "No symptoms provided"}), 400

    # Convert symptoms to model input (example: one-hot encoding)
    input_vector = [1 if s in symptoms else 0 for s in all_possible_symptoms]  # predefined list
    prediction = model.predict([input_vector])[0]
    
    return jsonify({"prediction": prediction})

@app.route('/')
def home():
    return "AI Symptom Checker Running"

if __name__ == '__main__':
    app.run(debug=True)
