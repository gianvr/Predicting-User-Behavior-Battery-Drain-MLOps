import boto3
import os
from dotenv import load_dotenv

load_dotenv()

bucket_name = "mlops-dvc-final-user-behavior"

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
)


response = s3.list_objects_v2(Bucket=bucket_name)
if "Contents" in response:
    for obj in response["Contents"]:
        s3.delete_object(Bucket=bucket_name, Key=obj["Key"])

s3.delete_bucket(Bucket=bucket_name)