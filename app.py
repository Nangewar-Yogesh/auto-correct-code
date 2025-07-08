from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load_model('model/saved_model/seq2seq_model.h5')

# Dummy tokenizer example (replace with your actual tokenizer logic)
def preprocess_input(code):
    # Convert string to integer sequences, pad if needed
    return np.array([[1, 2, 3, 4]])  # Dummy data

def decode_prediction(pred):
    return "corrected code output"  # Replace with actual decoding

@app.route('/')
def home():
    return "✅ Auto-Correct ML App is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    buggy_code = data.get('code', '')

    # Preprocess
    input_seq = preprocess_input(buggy_code)

    # Predict
    prediction = model.predict(input_seq)

    # Decode prediction
    corrected_code = decode_prediction(prediction)

    return jsonify({
        'buggy_code': buggy_code,
        'corrected_code': corrected_code
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
