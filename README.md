# 🚗 Car Price Prediction using Logistic Regression

This project predicts **car prices** based on various features such as year, mileage, engine size, and more.  
It is integrated with **MLflow** for model tracking and versioning, and deployed as a **Flask web app**.  
A **GitHub Actions CI/CD pipeline** automates testing, model registration, and deployment on every commit.

---

## 🌐 Live Demo

- **Flask App:** [st125986.ml.brain.cs.ait.ac.th](https://st125986.ml.brain.cs.ait.ac.th/)
- **GitHub Repository:** [github.com/saugatshakya/predicting_car_price-3](https://github.com/saugatshakya/predicting_car_price-3)

---

## 📊 Project Overview

This project demonstrates an end-to-end **MLOps workflow**:
- Building a logistic regression model for car price prediction.
- Tracking experiments and model versions using **MLflow**.
- Automating unit tests and deployment with **GitHub Actions**.
- Hosting the Flask application that dynamically loads the **latest model** from MLflow.

---

## ⚙️ Tech Stack

| Component | Technology |
|------------|-------------|
| Model | Logistic Regression |
| Tracking | MLflow |
| Backend | Flask |
| CI/CD | GitHub Actions |
| Deployment | Brain Cluster (AIT ML Server) |
| Language | Python 3.9+ |

---

## 🚀 Workflow

```text
        🧑‍💻 Developer
              │
              ▼
     📂 GitHub Repository
              │  (Push Code)
              ▼
   ✅ GitHub Actions (CI/CD Pipeline)
        • Run Unit Tests
        • Train + Log Model to MLflow
              │
              ▼
      📊 MLflow Model Registry
        (Latest Version Stored)
              │
              ▼
   🌐 Flask App (Live Deployment)
   https://st125986.ml.brain.cs.ait.ac.th/
