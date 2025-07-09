from keras.models import load_model
model = load_model('model/saved_model/seq2seq_model1.h5')  # adjust path if needed
from flask import request, jsonify

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.json['text']  # Input from user
    # do preprocessing and prediction using your model
    prediction = model.predict(…)  # Your logic here
    return jsonify({'corrected': prediction})
