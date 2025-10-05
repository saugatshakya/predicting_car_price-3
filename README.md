# 🚗 Car Price Prediction using MLflow & CI/CD

This project predicts **used car prices** using a trained machine learning model integrated with **MLflow**, **Flask**, and a fully automated **CI/CD pipeline**.
Once new code is pushed to GitHub, the pipeline automatically runs the model workflow, logs the new model to MLflow, and redeploys the Flask app using the latest version.

---

## 🌐 Live Demo

🔗 **App:** [https://st125986.ml.brain.cs.ait.ac.th/](https://st125986.ml.brain.cs.ait.ac.th/)
💻 **GitHub Repo:** [https://github.com/saugatshakya/predicting_car_price-3](https://github.com/saugatshakya/predicting_car_price-3)

---

## 🧠 Key Features

* **Machine Learning Model** for predicting car prices
* **MLflow Integration** for experiment tracking and model versioning
* **Automated CI/CD** pipeline using **GitHub Actions**
* **Flask API Deployment** that dynamically uses the **latest MLflow model**
* **Dockerized Setup** for fast and consistent local deployment
* **Automatic Redeployment** triggered on every successful commit

---

## ⚙️ Tech Stack

* **Python 3.10+**
* **Flask** – Web Framework
* **Scikit-learn** – Model Training
* **MLflow** – Experiment Tracking & Model Registry
* **GitHub Actions** – Continuous Integration / Deployment
* **Docker + Docker Compose** – Containerization

---

## 🧩 Project Structure

```
predicting_car_price-3/
│
├── app/
│   ├── main.py                # Flask app serving latest MLflow model
│   ├── model/                 # (optional) Local model storage
│
├── src/
│   ├── logistic_regression.py # Model training and MLflow logging
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml          # CI/CD pipeline definition
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 🚀 Running Locally (Dockerized)

Everything is containerized — no manual setup required.

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/saugatshakya/predicting_car_price-3.git
cd predicting_car_price-3
```

### 2️⃣ Start the Application

```bash
docker compose up --build
```

This will:

* Build the Docker image
* Launch the Flask API
* Automatically fetch and serve the **latest model** from MLflow

Once ready, open your browser and visit:

```
http://localhost:5000
```

---

## 🔄 CI/CD Workflow Overview

Each time you push code to GitHub:

1. **GitHub Actions** automatically builds and runs the containerized environment.
2. The workflow:

   * Trains or updates the ML model
   * Logs the new model to **MLflow**
   * Deploys the **Flask app** with the latest version
3. The live app always serves predictions using the newest validated model.

---

## 📈 Model Tracking with MLflow

* Each model version and its metrics are logged to MLflow.
* The Flask API automatically fetches the **latest production-ready model** at runtime.
* This ensures seamless, up-to-date inference without manual redeployment.

---

## 👨‍💻 Author

**Saugat Shakya**
Founding Member at **Galli Maps** | AR & AI Enthusiast
📧 [LinkedIn](https://www.linkedin.com/in/saugatshakya)

