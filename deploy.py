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
# run_name= "marvelous-flea-997"
# experiment_name= "Default"

print("run_name:", run_name)
# s3_bucket_name="dts-textract-test"
# s3_folder_name="mlflow_demo_models"
# Set the S3 bucket and folder where you want to store the artifacts

s3_bucket_name = os.environ['S3_BUCKET_NAME']
s3_folder_name = os.environ['S3_FOLDER_NAME']
runs = mlflow.search_runs(experiment_names=[experiment_name])

print(runs)
# Set up credentials for MLflow tracking server access

mlflow_tracking_uri= os.environ['MLFLOW_TRACKING_URI']
# mlflow_tracking_uri="http://127.0.0.1:5000"
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
print(run_id)



# Get the run's model URI

model_uri = f"runs:/{run_id}/model"

print(model_uri)


# model_name= "Car_price_prediction_model"

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




# client = mlflow.tracking.MlflowClient()

# model_version_details = client.get_model_version(model_name, model_version.version)
# print(model_version_details)
# artifact_path = model_version_details.source

# print(artifact_path)




#Download the artifact in local machine
run_id = mlflow.active_run().info.run_id
artifact_uri = os.path.join(mlflow.get_tracking_uri(), "mlruns", run_id, "artifacts")
dst_path = "C:/Users/v/Desktop/models"
model_path=mlflow.artifacts.download_artifacts(artifact_uri, dst_path)

# run_id = "012f572be865428c9ab5b701ab2a6d1c"  # specify the ID of the run that contains the artifacts you want to download
# run = mlflow.get_run(run_id)

# artifact_uri = run.info.artifact_uri


# local_dir = "C:\\Users\v\Desktop\models"  # specify the path of the local directory where you want to save the downloaded artifacts
# model_uri = "models:/Car_price_prediction_model/8"
# artifact_uri = mlflow.get_artifact_uri(model_uri)

# model_path = mlflow.artifacts.download_artifacts(artifact_uri, local_dir)

# model_path=mlflow.artifacts.download_artifacts(run_id=run_id, dst_path=local_dir)

# model_path = mlflow.artifacts.download_artifacts(artifact_uri, dst_path="C://Users/v/Desktop/models")

# use the downloaded model
# model_path = mlflow.artifacts.download_artifacts(artifact_path,dst_path="C:\\Users\v\Desktop\models")

print("MODEL PATH:",model_path)

# ## USING BOTO3 ##

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