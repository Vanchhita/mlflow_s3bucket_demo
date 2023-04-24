import os
import subprocess
import mlflow
import boto3
from mlflow import MlflowClient
import mlflow.exceptions
import requests
# # Set the experiment name ,run name, aws_key and aws_key_id

experiment_name = os.environ["EXPERIMENT_NAME"]

run_name = os.environ["RUN_NAME"]

access_key = os.environ['AWS_ACCESS_KEY_ID']

secret_key = os.environ['AWS_SECRET_ACCESS_KEY']

# experiment_name= "Default"

# run_name="wise-foal-997"
# print("run_name:", run_name)
# Set the S3 bucket and folder where you want to store the artifacts

s3_bucket_name = os.environ['S3_BUCKET_NAME']
s3_folder_name = os.environ['S3_FOLDER_NAME']
runs = mlflow.search_runs(experiment_names=[experiment_name])

print(runs)
# Set up credentials for MLflow tracking server access

mlflow_tracking_uri= os.environ['MLFLOW_TRACKING_URI']

print("MLFLOW_TRACKING_URI",mlflow_tracking_uri)

mlflow.set_tracking_uri(mlflow_tracking_uri)





# Get the run by name

runs = mlflow.search_runs(experiment_ids=mlflow.get_experiment_by_name(experiment_name).experiment_id,
                          filter_string=f"tags.mlflow.runName = '{run_name}'")

if len(runs) == 0:
    raise ValueError(f"No runs found with name {run_name} in experiment {experiment_name}")

elif len(runs) > 1:
    raise ValueError(f"Multiple runs found with name {run_name} in experiment {experiment_name}. Please use a unique name.")

run_id = runs.iloc[0].run_id




# Get the run's model URI

model_uri = f"runs:/{run_id}/model"

print(model_uri)




# Register the model

model_name = os.environ['MODEL_NAME']

model_version = mlflow.register_model(model_uri, model_name)




# Change the run's stage to "staging"

mlflow_client = mlflow.tracking.MlflowClient()

mlflow_client.transition_model_version_stage(
    name=model_name,version=model_version.version,stage="staging"

)




# Load the registered model artifact from the registry

model_uri = f"models:/{model_name}/{model_version.version}"

print("MODEL URI:",model_uri)




client = mlflow.tracking.MlflowClient()

model_version_details = client.get_model_version(model_name, model_version.version)

artifact_path = model_version_details.source

print(artifact_path)




#Download the artifact in local machine

model_path = mlflow.artifacts.download_artifacts(artifact_path,dst_path="C:\Users\v\Desktop\mlflow_demo\mlflow_s3bucket_demo\mlruns\models")

print("MODEL PATH:",model_path)

## USING BOTO3 ##

s3_client = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)
try:
# Upload the local directory to S3
    for root, dirs, files in os.walk(model_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            print("local_file_path:",local_file_path)
            s3_client.upload_file(local_file_path, s3_bucket_name,f"{s3_folder_name}/{model_name}/{model_version.version}/")




    s3_uri = "s3://{}/{}/{}/Version:{}".format(s3_bucket_name, s3_folder_name,model_name,model_version.version)

    print("Model artifacts stored in S3:", s3_uri)

except KeyError as e:

    print("Error while pushing to S3 bucket:",e)