import boto3
from botocore.client import Config
import pickle
from minio import Minio
from minio.error import S3Error
import urllib3
import pandas as pd

def model_predict(model_name, model_data):

    
    http_client = urllib3.PoolManager(
        timeout=urllib3.Timeout.DEFAULT_TIMEOUT,
        cert_reqs='CERT_NONE',
        retries=urllib3.Retry(
            total=5,
            backoff_factor=0.2,
            status_forcelist=[500, 502, 503, 504]
        )
    )


    minioClient = Minio(
    "127.0.0.1:9000",
    access_key="LjYOcfvfYyfYPg0ea3D3",
    secret_key="QKd4F1cgxMTLAh2MFtHYTWePbrurXNeMlf13h06D",
    secure=False  # Set to True if using https, False otherwise
    )

    try:
        # The bucket you want to access
        bucket_name = "ainexusmodels"
        
        # The name of the object (file) you want to download from the bucket
        object_name = f"{model_name}"
        
        # The local file path where you want to save the downloaded file
        file_path = f"../{model_name}"
        
        # Get the object
        response = minioClient.get_object(bucket_name, object_name)
        
        # Write the response content to a file
        with open(file_path, "wb") as file_data:
            for data in response.stream(32*1024):
                file_data.write(data)
                
        print(f"File {object_name} downloaded successfully.")
    except S3Error as exc:
        print("Error occurred:", exc)
    
    
    model_path = f'../{model_name}'
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    feature_names = ["age", "anaemia", "creatinine_phosphokinase", "diabetes", "ejection_fraction", "high_blood_pressure", "platelets", "serum_creatinine", "serum_sodium", "sex", "smoking", "time"]
    input_data = model_data["param_array"]
    input_df = pd.DataFrame([input_data], columns=feature_names)
    prediction = model.predict(input_df)
    return prediction