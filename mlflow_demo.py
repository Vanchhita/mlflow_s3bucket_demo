
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn import metrics 
import os
# from statistics import LinearRegression
import warnings
import sys
import os
import subprocess
import mlflow
import boto3
from mlflow import MlflowClient
import mlflow.exceptions
import requests
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow


import logging
# mlflow_uri="http://127.0.0.1:5000"
# mlflow.set_tracking_uri(mlflow_uri)
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2
# # Set the experiment name ,run name, aws_key and aws_key_id
experiment_name = os.environ["EXPERIMENT_NAME"]
run_name = os.environ["RUN_NAME"]
access_key = os.environ['AWS_ACCESS_KEY_ID']
secret_key = os.environ['AWS_SECRET_ACCESS_KEY']


# Set the S3 bucket and folder where you want to store the artifacts
s3_bucket_name = os.environ['S3_BUCKET_NAME']
s3_folder_name = os.environ['S3_FOLDER_NAME']
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    
    
    try:
        
        data = pd.read_csv('daTA.csv')
        
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )
    # print(data)  
    data.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True) 
    data.replace({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True) 
    data.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True) 
    X = data.drop(['Car_Name','Selling_Price'],axis=1) 
    Y = data['Selling_Price'] 
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=42) 
    # alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    alpha=0.5
    # l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    l1_ratio=0.9

    

# # Set the artifact URI to the new location
#     new_artifact_location = "0/012f572be865428c9ab5b701ab2a6d1c/artifacts"
#     mlflow.set_tracking_uri("http://localhost:5000")
#     experiment_id = 0  # Replace with the actual experiment ID
#     with mlflow.start_run(experiment_id=experiment_id):
    # Your code here

    with mlflow.start_run():
        #loading the linear regression model 
        lr = LinearRegression() 
        #Now we can fit the model to our dataset 
        lr.fit(X_train,Y_train) 
        # prediction on Training data 
        test_data_prediction = lr.predict(X_test) 
        (rmse, mae, r2) = eval_metrics(Y_test, test_data_prediction)
        print("Model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        # # Model registry does not work with file store
        # if tracking_url_type_store != "file":
        #     # Register the model
            
        #     mlflow.sklearn.log_model(lr, "model", registered_model_name="Car_price_prediction_Model")
        # else:
        #     mlflow.sklearn.log_model(lr, "model")
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
        name=model_name,
            version=model_version.version,
        stage="staging"
        )
        try:
            experiment = mlflow.get_experiment_by_name('shivering-dolphin-46')
            experiment_id = experiment.experiment_id
        except AttributeError:
            experiment_id = mlflow.create_experiment('shivering-dolphin-46', artifact_location='s3://dts-textract-test/mlflow_demo_models/')

# Load the registered model artifact from the registry
        model_uri = f"models:/{model_name}/{model_version.version}"
        print("MODEL URI:",model_uri)

        client = mlflow.tracking.MlflowClient()
        model_version_details = client.get_model_version(model_name, model_version.version)
        artifact_path = model_version_details.source
        print(artifact_path)

#Download the artifact in local machine
        model_path = mlflow.artifacts.download_artifacts(artifact_path,dst_path=os.environ['ARTIFACT_DESTINATION_PATH'])
# model_path = mlflow.artifacts.download_artifacts(run_id, dst_path=os.environ['ARTIFACT_DESTINATION_PATH'])
        print("MODEL PATH=",model_path)


## USING BOTO3 ##

        s3_client = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)
        try:
        # Upload the local directory to S3
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    local_path = os.path.join(root, file)
                    s3_path = os.path.join(f"{s3_folder_name}/{model_name}/Version-{model_version.version}", local_path[len(model_path)+1:])
                    print(s3_path)
                    s3_client.upload_file(local_path,s3_bucket_name, s3_path)

            s3_uri = "s3://{}/{}/{}/Version:{}".format(s3_bucket_name, s3_folder_name,model_name,model_version.version)
            print("Model artifacts stored in S3:", s3_uri)
        except KeyError as e:
            print("Error while pushing to S3 bucket:",e)

