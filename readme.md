# 🚗 Car Price Prediction Web App

A Flask-based machine learning web application for predicting car prices, containerized with Docker. The project uses a trained Random Forest Regressor model built from a dataset of 8,128 cars with features like year, km_driven, mileage, engine, max_power, and brand.

## Demo
[Live Demo](https://carprice-predict.ambitiousisland-1be3b1ed.southeastasia.azurecontainerapps.io/)

![Python](https://img.shields.io/badge/python-v3.9+-blue.svg)
![Flask](https://img.shields.io/badge/flask-v2.0+-green.svg)
![Docker](https://img.shields.io/badge/docker-supported-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

- **High-Accuracy ML Model**: Random Forest Regressor with R² ≈ 0.90 trained on preprocessed car dataset
- **User-Friendly Web Interface**: Responsive frontend for inputting car details and receiving price predictions
- **Docker Support**: Containerized for seamless deployment across platforms
- **Simple Input Parameters**: Year, km_driven, mileage, engine, max_power, brand
- **API Endpoint**: RESTful API for integration with other applications
- **Fuel Efficiency Integration**: Incorporates mileage for enhanced prediction accuracy

## 📂 Project Structure

```
.
├── app/
│   ├── main.py              # Flask application entry point
│   ├── model/
│   │   └── car_price_model.pkl  # Trained Random Forest model
│   ├── templates/
│   │   └── index.html      # Main webpage for user input
│   ├── Dockerfile          # Docker image configuration
│   └── requirements.txt    # Python dependencies
├── main.ipynb              # EDA, data processing and model training notebook
└── README.md              # Project documentation
```

````

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Docker (optional, for containerized deployment)

### 1. Clone the Repository

```bash
git clone https://github.com/saugatshakya/predicting_car_price.git
cd car-price-prediction-app
cd app
````

### 2. Run Locally (without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

The web app will be available at: **http://127.0.0.1:5001**

### 3. Run with Docker

```bash
# Build the Docker image
docker build -t car-price-app .

# Run the container
docker run -p 5001:5000 car-price-app
```

Visit: **http://127.0.0.1:5001** to access the web app.

## 🌐 Usage

### Web Interface

1. Open the web app in your browser (http://127.0.0.1:5001)
2. Enter car details in the provided form:
   - **Year**: Manufacturing year of the car
   - **KM Driven**: Total kilometers driven
   - **Mileage**: Fuel efficiency (kmpl)
   - **Engine**: Engine capacity (CC)
   - **Max Power**: Maximum power (bhp)
   - **Brand**: Car brand name
3. Submit to receive the predicted car price

### API Endpoint

**Endpoint**: `/predict`  
**Method**: `POST`  
**Content-Type**: `application/json`

#### Request Example

```json
{
  "year": 2018,
  "km_driven": 35000,
  "mileage": 23.4,
  "engine": 1497,
  "max_power": 118,
  "brand": "Honda"
}
```

#### Response Example

```json
{
  "price": 925000
}
```

## 📊 Model Performance

| Metric            | Value                                                  |
| ----------------- | ------------------------------------------------------ |
| **Model Type**    | Random Forest Regressor                                |
| **R² Score**      | ~0.90                                                  |
| **Dataset Size**  | 8,128 cars                                             |
| **Features Used** | 6 (year, km_driven, mileage, engine, max_power, brand) |

### Feature Importance Analysis

| Feature       | Correlation | RF Importance | Impact                                    |
| ------------- | ----------- | ------------- | ----------------------------------------- |
| **Year**      | 0.718       | ~0.518        | Newer cars command higher prices          |
| **Max Power** | 0.637       | ~0.335        | Stronger engines indicate premium models  |
| **Engine**    | 0.468       | ~0.078        | Larger engines suggest luxury/performance |
| **Mileage**   | 0.152       | ~0.043        | Higher efficiency slightly boosts value   |
| **Brand**     | -0.018      | ~0.025        | Premium brand perception                  |
| **KM Driven** | -0.185      | ~0.020        | Higher mileage reduces price              |

### Model Comparison

| Model                     | R² Score   |
| ------------------------- | ---------- |
| **Random Forest**         | **0.9016** |
| K-Neighbors (KNN)         | 0.8831     |
| Support Vector Regression | 0.8713     |
| Decision Tree             | 0.8411     |
| Linear Regression         | 0.8239     |

## 🔍 Data Preprocessing

The dataset underwent comprehensive preprocessing:

- **Owner Mapping**: First Owner → 1, Second Owner → 2, etc.
- **Fuel Type Filtering**: Removed CNG and LPG cars due to different mileage units
- **Data Cleaning**: Stripped units from numeric columns ("kmpl", "CC")
- **Brand Extraction**: Extracted brand from car names
- **Feature Selection**: Removed inconsistent features (torque, test drive cars)
- **Price Transformation**: Log-transformed selling price for stability

## 🛠 Tech Stack

- **Backend**: Python 3.9+, Flask
- **Machine Learning**: Scikit-learn, Pandas, NumPy
- **Frontend**: HTML, CSS, JavaScript
- **Containerization**: Docker
- **Model**: Random Forest Regressor

## 🚀 Deployment Options

The application can be deployed to:

- **Local**: Docker container (default)
- **Cloud Platforms**:
  - AWS ECS/EC2
  - Google Cloud Run
  - Azure App Service
  - Heroku/Render
- **Integration**: Can be integrated into larger web/mobile ecosystems

## 📌 Future Improvements

- [ ] Add support for additional features (transmission, fuel type, seats)
- [ ] Enhance preprocessing for unseen car brands
- [ ] Deploy to public cloud endpoint
- [ ] Implement unit tests and CI/CD pipeline
- [ ] Improve web interface with advanced styling
- [ ] Add real-time model retraining capabilities
- [ ] Include confidence intervals for predictions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Chaky's Company Team**

- 📧 Email: saugatoff@gmail.com
- 🐙 GitHub: [@saugatshakya](https://github.com/saugatshakya)

## 🙏 Acknowledgments

- Dataset contributors and the open-source community
- Flask and Scikit-learn documentation and tutorials
- Docker community for containerization best practices

---

⭐ **If you found this project helpful, please give it a star!**
