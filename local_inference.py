import os
import mlflow
import mlflow.pyfunc
import pandas as pd
import pickle
import numpy as np
from app.src.logistic_regression import *
import joblib

# Now you can pass raw Pandas DataFrame/Series directly
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


# Load from MLflow registry
model_uri = "app/model/st125986-a3-model_j.pkl"
loaded_model = joblib.load(model_uri)

prediction = loaded_model.predict(sample_df)
print("Predicted price:", prediction)