import pickle
import numpy as np
from flask import Flask, request, jsonify

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return "Breast Cancer Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']  # Expecting JSON {"features": [..list of values..]}
    scaled_data = scaler.transform([data])
    prediction = model.predict(scaled_data)
    return jsonify({'prediction': int(prediction[0])})  # 0 = Benign, 1 = Malignant

if __name__ == '__main__':
    app.run(debug=True)
