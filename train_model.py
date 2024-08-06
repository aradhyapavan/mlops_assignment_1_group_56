from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import mlflow
import data_preprocessing

# Load processed data
X = data_preprocessing.X
y = data_preprocessing.y

# Split the data
file_path = 'train_output'
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

# Set up MLflow tracking
mlflow.set_tracking_uri('http://127.0.0.1:5000/')
mlflow.autolog()

experiment_name = "liver_disease_prediction"
mlflow.set_experiment(experiment_name)

with mlflow.start_run():
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Log the model
    mlflow.sklearn.log_model(model, 'model')
    
    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('n_estimators', model.n_estimators)

# Save the training and test data
X_train.to_csv(file_path + '/X_train.csv')
y_train.to_csv(file_path + '/Y_train.csv')
X_test.to_csv('test_output/X_test.csv')
y_test.to_csv('test_output/Y_test.csv')

# Save the model
dump(model, 'models/model.joblib')
