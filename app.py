from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    code = data.get('code')
    
    # dummy example
    corrected_code = code.replace("prnt", "print")
    
    return jsonify({'corrected_code': corrected_code})
