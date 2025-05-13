from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Load model and scaler
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = data.get('features')

        if not features or len(features) != 30:
            return jsonify({'error': 'Exactly 30 features are required.'}), 400

        # Reshape and scale
        input_array = np.array(features).reshape(1, -1)
        scaled_input = scaler.transform(input_array)
        prediction = model.predict(scaled_input)[0]

        result = "Malignant (Cancerous)" if prediction == 0 else "Benign (Non-Cancerous)"

        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)