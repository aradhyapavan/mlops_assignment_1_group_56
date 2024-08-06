from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import pandas as pd
import data_preprocessing
import mlflow
import os

# Load processed data
X = data_preprocessing.X
y = data_preprocessing.y

# Define output directories
file_path = 'train_output'
output_file_path = 'test_output'
model_dir = 'models'

# Ensure the directories exist
os.makedirs(file_path, exist_ok=True)
os.makedirs(output_file_path, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

# Set up MLflow tracking
mlflow.set_tracking_uri('http://127.0.0.1:5000/')
mlflow.set_experiment(experiment_name="my_experiment")  # Better to use a name instead of ID
mlflow.autolog()

# Train and log the model
with mlflow.start_run():
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Log the model
    mlflow.sklearn.log_model(model, 'model')
    
    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('n_estimators', model.n_estimators)

# Save the train and test data
X_train.to_csv(os.path.join(file_path, 'X_train.csv'), index=False)
y_train.to_csv(os.path.join(file_path, 'Y_train.csv'), index=False)
X_test.to_csv(os.path.join(output_file_path, 'X_test.csv'), index=False)
y_test.to_csv(os.path.join(output_file_path, 'Y_test.csv'), index=False)

# Save the model
dump(model, os.path.join(model_dir, 'model.joblib'))
