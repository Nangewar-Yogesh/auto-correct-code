from flask import Flask, request, render_template
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json


app = Flask(__name__)

# Load model and tokenizer
model = load_model('model/saved_model/seq2seq_model.h5')
with open('model/saved_model/tokenizer.json', 'r') as f:
    tokenizer_json = f.read()  # read as string
    tokenizer = tokenizer_from_json(tokenizer_json)  # pass string


MAX_LEN = 50
index_word = {v: k for k, v in tokenizer.word_index.items()}
index_word[0] = ''

def decode_sequence(seq):
    pred = model.predict(seq)[0]
    pred_words = [index_word.get(np.argmax(token), '') for token in pred]
    return ' '.join(pred_words).replace(' <OOV>', '').strip()

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    if request.method == "POST":
        buggy_code = request.form["buggy_code"]
        seq = tokenizer.texts_to_sequences([buggy_code])
        seq = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
        result = decode_sequence([seq[0]])
    return render_template("index.html", corrected_code=result)

if __name__ == "__main__":
    app.run(debug=True)
