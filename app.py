from flask import Flask, request, jsonify
import tensorflow as tf
from keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load_model("model/saved_model/seq2seq_model.h5")

# Dummy preprocessing and postprocessing for demonstration
def preprocess(code):
    # You should replace this with actual tokenizer/encoder logic
    return np.array([[1, 2, 3]])

def postprocess(prediction):
    # Replace this with your actual decoding logic
    return "for(int i=0; i<10; i++) {"

@app.route('/')
def home():
    return "✅ Auto-Correct ML App is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    buggy_code = data.get("buggy_code")

    if buggy_code is None:
        return jsonify({"error": "No buggy_code field provided"}), 400

    # Preprocess the input
    input_seq = preprocess(buggy_code)

    # Predict using the model
    prediction = model.predict(input_seq)

    # Postprocess to get readable output
    corrected_code = postprocess(prediction)

    return jsonify({
        "buggy_code": buggy_code,
        "corrected_code": corrected_code
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
