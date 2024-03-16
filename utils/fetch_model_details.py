from minio import Minio
from minio.error import S3Error



def fetch_model_details(model_name):
    bucket_name = "ainexusmetrics"

    minio_client = Minio(
            "127.0.0.1:9000",
            access_key="LjYOcfvfYyfYPg0ea3D3",
            secret_key="QKd4F1cgxMTLAh2MFtHYTWePbrurXNeMlf13h06D",
            secure=False  # Set to True if using HTTPS
    )

    try:
        # Fetch the data from the MinIO bucket
        data = minio_client.get_object(bucket_name, model_name)

        # Read and print the content of the object
        content = data.read().decode('utf-8')
        return {"msg":"Success", "code":200, "content":content}

    except S3Error as e:
        if e.code == "NoSuchKey":
            return {"msg":"Model not found", "code":404}
        else:
            print(f"Error fetching data from MinIO: {e}")
            return {"msg":"There was an error while fetching data", "code":500}

