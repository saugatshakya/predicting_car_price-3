import mlflow
import pickle
import pandas as pd
from app.model.st125986_a3_model import CarPricePredictor

# Load local model
with open("app/model/st125986-a3-model.pkl", "rb") as f:
    predictor = pickle.load(f)

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

with mlflow.start_run(run_name="logistic_regression_deploy"):
    mlflow.pyfunc.log_model(
        name="st125986-a3-model",
        python_model=CarPriceWrapper(predictor),
        input_example=sample_df
    )

print("Model deployed to MLflow!")
