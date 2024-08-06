from sklearn.metrics import accuracy_score
from joblib import load
import pandas as pd
import mlflow

# Paths for loading and saving data
test_path = 'test_output'
output_path = 'final_output'

# Set up MLflow tracking
mlflow.set_tracking_uri('http://127.0.0.1:5000/')
mlflow.autolog()

experiment_name = "liver_disease_prediction"
mlflow.set_experiment(experiment_name)

# Load test data and model
X_test = pd.read_csv(test_path + '/X_test.csv')

# Remove the 'Unnamed: 0' column if it exists
if 'Unnamed: 0' in X_test.columns:
    X_test = X_test.drop(columns=['Unnamed: 0'])

model = load('models/model.joblib')

# Make predictions
y_pred = model.predict(X_test)

with mlflow.start_run():
    accuracy = accuracy_score(y_pred, model.predict(X_test))
    mlflow.log_metric('accuracy', accuracy)

# Convert y_pred to DataFrame
y_pred_df = pd.DataFrame(y_pred, columns=['Prediction'])

# Combine X_test and y_pred_df
combined_df = pd.concat([X_test, y_pred_df], axis=1)

# Save the combined DataFrame to a CSV file
combined_df.to_csv(output_path + '/final_predicted_output.csv', index=False)
