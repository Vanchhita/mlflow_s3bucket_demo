
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
    # experiment_id = mlflow.create_experiment('mlflow-demo-ex1', artifact_location='c:/Users/v/Desktop/mlflow_demo/mlflow_s3bucket_demo/mlruns/0/')
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
        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            
            mlflow.sklearn.log_model(lr, "model", registered_model_name="Car_price_prediction_Model")
        else:
            mlflow.sklearn.log_model(lr, "model")
        
        # try:
        #     experiment = mlflow.get_experiment_by_name('Default')
        #     experiment_id = experiment.experiment_id
        # except AttributeError:
        




