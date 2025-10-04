import mlflow
import pickle
import pandas as pd
from app.src.logistic_regression import *
import joblib
# Load local model

local_path = "app/model/st125986-a3-model.pkl"
predictor = joblib.load(local_path)
# Wrap model for MLflow
class CarPriceWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, predictor):
        self.predictor = predictor

    def predict(self, context, model_input: pd.DataFrame):
        return self.predictor.predict(model_input)

# Example input for MLflow schema
sample_df = pd.DataFrame([{
    'year': 2014,
    'km_driven': 145500,
    'fuel': 'Diesel',
    'seller_type': 'Individual',
    'transmission': 'Manual',
    'owner': 1,
    'mileage': 23.4,
    'engine': 1248.0,
    'max_power': 74.0,
    'seats': 5.0,
    'brand': 'Maruti',
    'brand_type': 'Mass-Market'
}])

mlflow.set_tracking_uri("https://mlflow.ml.brain.cs.ait.ac.th/")
mlflow.set_experiment("st125986-a3")

with mlflow.start_run(run_name="logistic_regression_deploy") as run:
    mlflow.pyfunc.log_model(
        name="model",
        python_model=CarPriceWrapper(predictor),
        input_example=sample_df
    )
    # Construct proper model URI to register
    model_uri = f"runs:/{run.info.run_id}/model"

# Register as a new version
registered_model = mlflow.register_model(
    model_uri=model_uri,
    name="st125986-a3-model"
)

print("Model deployed to MLflow!")
