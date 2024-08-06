from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the model
model_path = os.path.join(os.path.dirname(__file__), '../models/model.joblib')
model = joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from the request data
        data = request.get_json(force=True)
        features = np.array(data['features']).reshape(1, -1)
        
        # Make predictions
        prediction = model.predict(features)
        
        # Return the prediction as JSON
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
