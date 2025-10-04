import os
import logging
import joblib
import flask
import numpy as np
from werkzeug.exceptions import BadRequest
from linear_reg import LinearRegression
import pandas as pd
import mlflow


app = flask.Flask(__name__, template_folder='templates')

mlflow_tracking_url = os.environ.get("MLFLOW_TRACKING_URL", "https://mlflow.ml.brain.cs.ait.ac.th/")

mlflow.set_tracking_uri(mlflow_tracking_url)
mlflow.set_experiment("st125986-a3")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    model_data = joblib.load("model/car_price_model.pkl")
    new_model = joblib.load("model/custom_car_price_model.pkl")
    logger.info("✅ Model loaded successfully.")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    raise

def predict_car_price(input_data, model_data):
    """
    Predict car price based on input features.
    """
    model = model_data['model']
    scaler = model_data['scaler']
    le_dict = model_data['le_dict']
    feature_cols = model_data['feature_cols']

    processed_data = []
    for col in feature_cols:
        if col not in input_data:
            raise BadRequest(f"Missing required field: {col}")
        if col in le_dict:  # categorical
            try:
                encoded_val = le_dict[col].transform([input_data[col]])[0]
            except ValueError:
                logger.warning(f"Unseen label '{input_data[col]}' for column '{col}'. Using default=0.")
                encoded_val = 0
            processed_data.append(encoded_val)
        else:
            try:
                processed_data.append(float(input_data[col]))
            except (ValueError, TypeError):
                raise BadRequest(f"Invalid value for numerical field: {col}")

    sample = np.array(processed_data).reshape(1, -1)
    sample_scaled = scaler.transform(sample)

    pred_log_price = model.predict(sample_scaled)[0]
    return float(np.exp(pred_log_price))

# Create a prediction function for demonstration
def predict_car_price_v2(input_data, model_data):
    model = model_data['model']
    le_dict = model_data['le_dict']
    feature_cols = model_data['feature_cols']

    processed = []
    for col in feature_cols:
        if col in le_dict:
            try:
                processed.append(le_dict[col].transform([input_data[col]])[0])
            except ValueError:
                processed.append(0)
        else:
            processed.append(input_data[col])

    sample = np.array(processed).reshape(1, -1)
    # Predict log price
    pred_log_price = model.predict(pd.DataFrame(sample, columns=feature_cols))[0]
    # Convert back to actual price
    pred_price = np.exp(pred_log_price)
    return pred_price

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = flask.request.get_json(force=True)
        predicted_price = predict_car_price(data, model_data)
        return flask.jsonify({'price': round(predicted_price, 2)})
    except BadRequest as e:
        return flask.jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return flask.jsonify({"error": "Internal Server Error"}), 500


@app.route('/predict_v2', methods=['POST'])
def predict_v2():
    try:
        data = flask.request.get_json(force=True)
        predicted_price = predict_car_price_v2(data, new_model)
        return flask.jsonify({'price': round(predicted_price, 2)})
    except BadRequest as e:
        return flask.jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return flask.jsonify({"error": "Internal Server Error"}), 500

@app.route('/classify', methods=["POST"])
def classify():
    try:
        data = flask.request.get_json(force=True)
        model_uri = "models:/st125986-a3-model/latest"
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        df = pd.DataFrame([data])

        # Explicit dtype casting
        df = df.astype({
        "year": "int64",
        "km_driven": "int64",
        "owner": "int64",
        "mileage": "float64",
        "engine": "float64",
        "max_power": "float64",
        "seats": "float64",
        })
        prediction = loaded_model.predict(df)
        print(f"{prediction[0]}")
        return flask.jsonify({"class": int(prediction[0])})
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return flask.jsonify({"error": "Internal Server Error"}), 500

@app.route('/health', methods=['GET'])
def health():
    return flask.jsonify({"status": "ok"}), 200

@app.route('/')
def home():
    return flask.render_template('index.html')

if __name__ == '__main__':
    
    example_car = {
    'year': 2014,
    'km_driven': 145500,
    'engine': 1248,
    'max_power': 74,
    'brand': 'Maruti',
    'mileage': 23.4
}
    print("Example prediction:", predict_car_price(example_car, model_data))
    print("Example prediction v2:", predict_car_price_v2(example_car, new_model))
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port)  # no debug in prod
