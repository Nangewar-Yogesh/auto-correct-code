import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Load the model
model = load_model('model/saved_model/seq2seq_model.h5')

# Load tokenizer
with open('model/saved_model/tokenizer.json', 'r') as f:
    data = f.read()
tokenizer = tokenizer_from_json(data)


# Parameters
MAX_LEN = 50

# Reverse lookup for predictions
index_word = {v: k for k, v in tokenizer.word_index.items()}
index_word[0] = ''  # for padding

def decode_sequence(seq):
    pred = model.predict(seq)[0]
    pred_words = [index_word.get(np.argmax(token), '') for token in pred]
    return ' '.join(pred_words).replace(' <OOV>', '').strip()

def correct_code(buggy_code):
    input_seq = tokenizer.texts_to_sequences([buggy_code])
    input_seq = pad_sequences(input_seq, maxlen=MAX_LEN, padding='post')

    # Use the same sequence as both encoder and decoder input
    prediction = decode_sequence([input_seq[0]])
    return prediction

# Try it out
if __name__ == "__main__":
    while True:
        buggy = input("\n📝 Enter buggy code (or type 'exit'): ")
        if buggy.lower() == 'exit':
            break
        corrected = correct_code(buggy)
        print(f"✅ Corrected Code:\n{corrected}")
