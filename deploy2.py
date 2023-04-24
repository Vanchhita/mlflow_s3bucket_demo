import os
import boto3
import mlflow

# Set environment variables
experiment_name = os.environ["EXPERIMENT_NAME"]
run_name = os.environ["RUN_NAME"]
s3_bucket = os.environ['S3_BUCKET_NAME']
s3_folder = os.environ['S3_FOLDER_NAME']
aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Search for the MLflow run
runs = mlflow.search_runs(experiment_names=experiment_name, run_name=run_name)
if len(runs) == 0:
    raise ValueError(f"No matching run found for experiment {experiment_name} and run {run_name}")
run_id = runs.iloc[0].run_id

# Register the model in the MLflow model registry
model_uri = f"runs:/{run_id}/model"
model_details = mlflow.register_model(model_uri, "my_registered_model")

# Transition the model version to "staging" stage in the MLflow model registry
model_version = model_details.version
mlflow.models.transition_model_version_stage(
    name=model_details.name, version=model_version, stage="staging"
)

# Download the model artifacts to a local directory
artifact_path = f"model/{model_version}"
model_path = mlflow.artifacts.download_artifacts(artifact_path, dst_path="./Models")

# Use Boto3 to upload the local directory to an S3 bucket
s3_client = boto3.client(
    "s3",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)
s3_client.upload_file(model_path, s3_bucket, f"{s3_folder}/model_{model_version}")

# Print the S3 URI where the model artifacts are stored
s3_uri = f"s3://{s3_bucket}/{s3_folder}/model_{model_version}"
print(f"Model artifacts uploaded to {s3_uri}")
