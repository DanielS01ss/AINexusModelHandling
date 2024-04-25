import os
import uuid
import mlflow
import mlflow.sklearn
import shap
from minio import Minio, S3Error
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,  precision_score, recall_score, f1_score, roc_auc_score
from utils.save_model_for_user  import save_model_for_users
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import time
import pickle
import json


def train_and_store_random_forest(dataset, ml_params,email):
    uuid_for_model = uuid.uuid4()
    dataset = json.loads(dataset)
    df = pd.DataFrame()
    df = pd.json_normalize(dataset)

    target_var = ml_params["target"]
    X_data = df.drop(columns=[target_var])  # Use drop(columns=...) to remove the target column
    y_data = df[target_var]
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

    column_names_array = X_data.columns.to_numpy()
    # or as a list
    column_names_list = X_data.columns.tolist()
    now = datetime.now()
    # Format the date and time as a string in the desired format
    formatted_now = now.strftime("%d/%m/%Y %H:%M")

    # Train a RandomForest model
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        explainer = shap.TreeExplainer(model)

        # Compute SHAP Values for your test set
        shap_values = explainer.shap_values(X_test)
        plt.figure()
        shap.summary_plot(shap_values, X_test)  # SHAP plot function does not use a `title` argument
        plt.title("SHAP Feature Importance")  # Add title with Matplotlib
        fig = plt.gcf()
        # Optionally, if you want to save the figure to a file
        fig.savefig(f"{uuid_for_model}_shap_summary_plot.png")

        shap_values_single = explainer.shap_values(X_test.iloc[0])



        # Generate force plot for the first in stance and the class of interest
        # Adjust the index [1] if you have a multi-class scenario and are interested in a different class
        force_plot = shap.force_plot(explainer.expected_value[1], shap_values_single[1], X_test.iloc[0])

        # Display the force plot
        shap.save_html(f"{uuid_for_model}_force_plot.html", force_plot)  # Save the plot to an HTML file    
        

        
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

        #here we save the data for our model like the png file and the html file

        file_path = f"{uuid_for_model}_shap_summary_plot.png"  # Path to your local file
        bucket_name = 'model-shap-values'     # Your MinIO bucket
        object_name = f"{uuid_for_model}_shap_summary_plot.png"  # Object name in MinIO

        minio_client = Minio(
            "127.0.0.1:9000",
            access_key="LjYOcfvfYyfYPg0ea3D3",
            secret_key="QKd4F1cgxMTLAh2MFtHYTWePbrurXNeMlf13h06D",
            secure=False  # Set to True if using HTTPS
        )

        # Upload the file
        try:
            with open(file_path, "rb") as file_data:
                file_stat = os.stat(file_path)
                minio_client.put_object(
                    bucket_name,
                    object_name,
                    file_data,
                    file_stat.st_size,
                    content_type='image/png'
                )
            print(f"'{file_path}' is successfully uploaded as '{object_name}' to the bucket '{bucket_name}'.")
        except S3Error as exc:
            print("Error occurred:", exc)
        

        file_path = f"{uuid_for_model}_force_plot.html"  # Path to your local file
        bucket_name = 'model-shap-values'     # Your MinIO bucket
        object_name = f"{uuid_for_model}_force_plot.html"  # Object name in MinIO
        

        try:
            with open(file_path, "rb") as file_data:
                file_stat = os.stat(file_path)
                minio_client.put_object(
                    bucket_name,
                    object_name,
                    file_data,
                    file_stat.st_size,
                    content_type='text/html'
                )
            print(f"'{file_path}' is successfully uploaded as '{object_name}' to the bucket '{bucket_name}'.")
        except S3Error as exc:
            print("Error occurred:", exc)
        
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
        "metrics": {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "roc_auc": roc_auc},
        "columns_used":column_names_list,
        "date": formatted_now
        }
        
        model_pkl_path = "{}_random_forest_model.pkl".format(uuid_for_model)

        metadata_file_path = "{}_metadata.json".format(uuid_for_model)

        
        with open(metadata_file_path, "w") as metadata_file:
            json.dump(metadata, metadata_file)
        

        with open(model_pkl_path, "wb") as model_file:
            pickle.dump(model, model_file)

        
        was_saved_successfully = True
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
            was_saved_successfully = False
        finally:
            # Optionally, you can delete the local metadata file after uploading
            if was_saved_successfully == True:
                save_model_for_users(email, model_pkl_path)
            if os.path.exists(metadata_file_path):
                os.remove(metadata_file_path)
        
        return model_pkl_path

