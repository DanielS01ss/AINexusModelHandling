import os
import uuid
import mlflow
import mlflow.sklearn
from minio import Minio, S3Error
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,  precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import pandas as pd
import time
import pickle
import json


def train_and_store_random_forest(dataset, ml_params):
    dataset = json.loads(dataset)
    df = pd.DataFrame()
    df = pd.json_normalize(dataset)
    target_var = ml_params["target"]
    X_data = df.drop(columns=[target_var])  # Use drop(columns=...) to remove the target column
    y_data = df[target_var]

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

    # Train a RandomForest model
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        duration = end_time - start_time
        # Log the duration to MLflow
        mlflow.log_metric("training_duration_seconds", duration)

        # Make predictions on the test set
        y_pred = model.predict(X_test)


        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

        # Log the model and metrics with MLflow
        mlflow.sklearn.log_model(model, "random_forest_model")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        metadata = {
        "parameters": {"n_estimators": 100, "random_state": 42},
        "metrics": {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "roc_auc": roc_auc}
        }
        uuid_for_model = uuid.uuid4()
        model_pkl_path = "{}_random_forest_model.pkl".format(uuid_for_model)

        metadata_file_path = "{}_metadata.json".format(uuid.uuid4())

        



        with open(metadata_file_path, "w") as metadata_file:
            json.dump(metadata, metadata_file)
        


        with open(model_pkl_path, "wb") as model_file:
            pickle.dump(model, model_file)

        minio_client = Minio(
            "127.0.0.1:9000",
            access_key="LjYOcfvfYyfYPg0ea3D3",
            secret_key="QKd4F1cgxMTLAh2MFtHYTWePbrurXNeMlf13h06D",
            secure=False  # Set to True if using HTTPS
        )

        try:
            minio_client = Minio(
                "127.0.0.1:9000",
                access_key="LjYOcfvfYyfYPg0ea3D3",
                secret_key="QKd4F1cgxMTLAh2MFtHYTWePbrurXNeMlf13h06D",
                secure=False  # Set to True if using HTTPS
            )

            minio_client.fput_object(
                "ainexusmodels",
                model_pkl_path,
                model_pkl_path
            )

            minio_client.fput_object(
                "ainexusmetrics",
                metadata_file_path,
                metadata_file_path
            )

            print(f"Metadata file uploaded to MinIO: {metadata_file_path}")
            print(f"Model file uploaded to MinIO: {model_pkl_path}")

        except S3Error as e:
            print(f"Error uploading metadata file to MinIO: {e}")

        finally:
            # Optionally, you can delete the local metadata file after uploading
            if os.path.exists(metadata_file_path):
                os.remove(metadata_file_path)

        # Save the model directly to MinIO
        try:
            model_pkl_path = "{}_random_forest_model.pkl".format(uuid.uuid4())
            with open(model_pkl_path, "wb") as model_file:
                pickle.dump(model, model_file)

            minio_client.fput_object(
                "ainexusmodels",
                model_pkl_path,
                model_pkl_path
            )
            print(f"Model file uploaded to MinIO: {model_pkl_path}")

        except S3Error as e:
            print(f"Error uploading model file to MinIO: {e}")

        finally:
        # Optionally, you can delete the local model file after uploading
            if os.path.exists(model_pkl_path):
                os.remove(model_pkl_path)
