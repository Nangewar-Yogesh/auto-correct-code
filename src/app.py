from flask import Flask
app = Flask(__name__)

@app.route('/')
def home():
    return "✅ Auto-Correct ML App is Running!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

  from flask import Flask, request, jsonify
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

@app.route('/')
def home():
    return "✅ Auto-Correct ML App is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    buggy_code = data.get('code')

    # Load tokenizers and model
    with open('model/tokenizer_input.pkl', 'rb') as f:
        tokenizer_input = pickle.load(f)
    with open('model/tokenizer_output.pkl', 'rb') as f:
        tokenizer_output = pickle.load(f)

    model = load_model('model/saved_model/seq2seq_model.h5')

    # Preprocess
    seq = tokenizer_input.texts_to_sequences([buggy_code])
    seq = pad_sequences(seq, maxlen=50, padding='post')
    prediction = model.predict([seq, seq])
    pred_ids = np.argmax(prediction[0], axis=-1)
    corrected_code = ' '.join(tokenizer_output.index_word.get(i, '') for i in pred_ids)

    return jsonify({'corrected_code': corrected_code.strip()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

