# test_logistic_regression_model.py
import pytest
import pandas as pd
import joblib
from app.src.logistic_regression import CarPricePredictor, LogisticRegression

@pytest.fixture
def sample_input():
    """Sample input DataFrame for testing."""
    return pd.DataFrame([{
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

@pytest.fixture
def model():
    """Load the trained model."""
    model_uri = "app/model/st125986-a3-model_j.pkl"
    return joblib.load(model_uri)

def test_model_prediction_type(model, sample_input):
    """Test that the model prediction returns a numeric value."""
    prediction = model.predict(sample_input)
    
    # Check that prediction is a numpy array
    assert isinstance(prediction, (np.ndarray, list)), "Prediction should be a numpy array or list"
    
    # Check that it contains a numeric value
    assert np.issubdtype(type(prediction[0]), np.number), "Predicted value should be numeric"

def test_model_prediction_non_empty(model, sample_input):
    """Test that the model returns at least one prediction."""
    prediction = model.predict(sample_input)
    
    # Ensure prediction is not empty
    assert len(prediction) > 0, "Prediction should not be empty"
