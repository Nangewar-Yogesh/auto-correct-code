from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin support

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'code' not in data:
            return jsonify({'error': 'Missing "code" in request'}), 400

        code = data['code']
        
        # Example dummy correction
        corrected_code = code.replace("prnt", "print")

        return jsonify({'corrected_code': corrected_code}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
