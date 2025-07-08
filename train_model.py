import json
import numpy as np
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    buggy = [item['buggy_code'] for item in data]
    fixed = [item['corrected_code'] for item in data]
    return buggy, fixed

def preprocess_data(input_texts, target_texts, num_words=10000, max_len=50):
    tokenizer = Tokenizer(num_words=num_words, filters='', lower=False)
    tokenizer.fit_on_texts(input_texts + target_texts)

    input_seq = tokenizer.texts_to_sequences(input_texts)
    target_seq = tokenizer.texts_to_sequences(target_texts)

    input_seq = pad_sequences(input_seq, maxlen=max_len, padding='post')
    target_seq = pad_sequences(target_seq, maxlen=max_len, padding='post')

    return input_seq, target_seq, tokenizer

def build_seq2seq_model(vocab_size, max_len, embedding_dim=128, latent_dim=256):
    encoder_inputs = Input(shape=(max_len,))
    x = Embedding(vocab_size, embedding_dim)(encoder_inputs)
    encoder_outputs, state_h, state_c = LSTM(latent_dim, return_state=True)(x)

    decoder_inputs = Input(shape=(max_len,))
    y = Embedding(vocab_size, embedding_dim)(decoder_inputs)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=False)
    decoder_outputs = decoder_lstm(y, initial_state=[state_h, state_c])
    decoder_dense = Dense(vocab_size, activation='softmax')(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_dense)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def train():
    buggy, fixed = load_data('data/code_pairs.json')
    input_seq, target_seq, tokenizer = preprocess_data(buggy, fixed)

    vocab_size = len(tokenizer.word_index) + 1
    max_len = input_seq.shape[1]

    model = build_seq2seq_model(vocab_size, max_len)

    target_seq_output = np.expand_dims(target_seq, -1)

    model.fit([input_seq, target_seq], target_seq_output,
              batch_size=2, epochs=10, validation_split=0.2)

    os.makedirs('model/saved_model', exist_ok=True)
    model.save('model/saved_model/seq2seq_model.h5')

    with open('model/saved_model/tokenizer.json', 'w') as f:
        f.write(tokenizer.to_json())

if __name__ == "__main__":
    train()
