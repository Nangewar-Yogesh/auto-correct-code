from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import json
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load model and tokenizer
model = load_model('model/saved_model/seq2seq_model.h5')

with open('model/saved_model/tokenizer.json') as f:
    data = json.load(f)
    tokenizer = Tokenizer.from_json(data)

MAX_LEN = 50

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    buggy_code = data.get('buggy_code')

    if not buggy_code:
        return jsonify({'error': 'No input provided'}), 400

    input_seq = tokenizer.texts_to_sequences([buggy_code])
    input_seq = pad_sequences(input_seq, maxlen=MAX_LEN, padding='post')

    prediction = model.predict([input_seq, input_seq])
    predicted_seq = np.argmax(prediction, axis=-1)

    corrected_tokens = tokenizer.sequences_to_texts(predicted_seq)[0]
    return jsonify({'corrected_code': corrected_tokens.strip()})

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=10000)
