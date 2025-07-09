@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        buggy_code = data.get('code')

        # Load model and tokenizers
        with open('model/tokenizer_input.pkl', 'rb') as f:
            tokenizer_input = pickle.load(f)
        with open('model/tokenizer_output.pkl', 'rb') as f:
            tokenizer_output = pickle.load(f)

        model = load_model('model/saved_model/seq2seq_model.h5')

        # Preprocess input
        seq = tokenizer_input.texts_to_sequences([buggy_code])
        seq = pad_sequences(seq, maxlen=50, padding='post')
        prediction = model.predict([seq, seq])
        pred_ids = np.argmax(prediction[0], axis=-1)
        corrected_code = ' '.join(tokenizer_output.index_word.get(i, '') for i in pred_ids)

        return jsonify({'corrected_code': corrected_code.strip()})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

