import mlflow
import uuid
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle


def train_and_store_SVM(dataset, ml_params):
    df = pd.DataFrame()
    df = pd.json_normalize(dataset)
    target_var = ml_params["target"]
    X_data = df.drop(target_var, axis=1)
    y_data = df[target_var]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

    # Train an SVM model
    with mlflow.start_run():
        model = SVC(kernel='linear', C=1)  # Example SVM model with linear kernel
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Log the model and metrics with MLflow
        mlflow.sklearn.log_model(model, "svm_model")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("kernel", "linear")
        mlflow.log_param("C", 1)

        model_pkl_path = "{}_resnet_model.pkl".format(uuid.uuid4())
        with open(model_pkl_path, "wb") as model_file:
            pickle.dump(model, model_file)
