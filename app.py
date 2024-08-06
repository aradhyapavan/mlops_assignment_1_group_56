from flask import Flask, request, jsonify
from joblib import load
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = load('models/model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.json
        
        # Convert the JSON data to a DataFrame
        df = pd.DataFrame(data)
        
        # Make predictions
        predictions = model.predict(df)
        
        # Return the predictions as a JSON response
        return jsonify(predictions.tolist())
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
