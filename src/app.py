from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
import json

app = Flask(__name__)

# Load your trained model
model = load_model("model/saved_model/seq2seq_model.h5")

# Dummy preprocessing (you must replace this with your tokenizer logic)
def preprocess(code):
    # Example placeholder
    return np.array([[1, 2, 3, 4]])  # Dummy sequence

# Dummy postprocessing (replace with actual decoding logic)
def postprocess(prediction):
    return "corrected code: for(int i=0; i<10; i++) {"

@app.route('/')
def home():
    return "✅ Auto-Correct ML App is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    buggy_code = data.get("buggy_code", "")

    if not buggy_code:
        return jsonify({"error": "No buggy_code provided"}), 400

    # Preprocess input
    input_seq = preprocess(buggy_code)

    # Predict using the model
    prediction = model.predict(input_seq)

    # Decode prediction
    corrected_code = postprocess(prediction)

    return jsonify({
        "buggy_code": buggy_code,
        "corrected_code": corrected_code
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
