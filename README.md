# Credit Risk Assessment

An end-to-end machine learning application that predicts the likelihood of loan default using borrower and loan-related information. The project covers data preprocessing, exploratory data analysis, model development, hyperparameter optimization, model explainability, error analysis, and deployment through FastAPI and Render.

**Live Application:** [Credit Risk Assessment](https://credit-risk-assessment-fix1.onrender.com)

**API Documentation:** [Swagger UI](https://credit-risk-assessment-fix1.onrender.com/docs)

**GitHub Repository:** [Credit Risk Assessment](https://github.com/Adithya-DSAIML/credit-risk-assessment)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Dataset and Features](#dataset-and-features)
4. [Project Workflow](#project-workflow)
5. [Model Development and Comparison](#model-development-and-comparison)
6. [Hyperparameter Optimization](#hyperparameter-optimization)
7. [Model Evaluation](#model-evaluation)
8. [Model Explainability with SHAP](#model-explainability-with-shap)
9. [Error Analysis](#error-analysis)
10. [Application Architecture](#application-architecture)
11. [Technology Stack](#technology-stack)
12. [Project Structure](#project-structure)
13. [Run Locally](#run-locally)
14. [API Usage](#api-usage)
15. [Deployment](#deployment)
16. [Limitations and Disclaimer](#limitations-and-disclaimer)

---

## Project Overview

Credit risk assessment is an important problem in financial institutions, where lenders need to evaluate the likelihood that a borrower will fail to repay a loan.

This project develops a binary classification model to estimate the probability of loan default based on borrower demographics, financial information, loan characteristics, and credit history.

The project goes beyond simply training a machine learning model. It implements a complete workflow:

- Exploratory Data Analysis and data cleaning.
- Feature preprocessing and transformation.
- Training and comparison of seven classification algorithms.
- Hyperparameter optimization using Optuna.
- Classification threshold optimization.
- Model evaluation and probability calibration.
- Model interpretation using SHAP.
- Error analysis of false positives and false negatives.
- Model deployment through FastAPI.
- Interactive frontend using HTML, CSS, and JavaScript.
- Deployment on Render.

The final application allows users to enter loan application details and receive a predicted default probability and classification.

## Problem Statement

The objective is to predict whether a borrower is likely to default on a loan.

This is formulated as a binary classification problem:

| Target | Meaning |
|---|---|
| 0 | Non-default |
| 1 | Default |

Instead of relying exclusively on a classification label, the application also returns the predicted probability of default.

A classification threshold is then applied to convert the predicted probability into a binary prediction.

## Dataset and Features

The project uses a loan risk dataset containing borrower and loan-related attributes.

The original dataset contains 32,581 records and 12 columns, including the target variable.

### Features used

| Feature | Description |
|---|---|
| `person_age` | Borrower's age |
| `person_income` | Annual income |
| `person_home_ownership` | Home ownership status |
| `person_emp_length` | Employment length in years |
| `loan_intent` | Purpose of the loan |
| `loan_grade` | Loan grade |
| `loan_amnt` | Loan amount |
| `loan_int_rate` | Loan interest rate |
| `loan_percent_income` | Loan amount as a proportion of income |
| `cb_person_default_on_file` | Previous default history |
| `cb_person_cred_hist_length` | Length of credit history |

The target variable is `loan_status`.

### Data preprocessing

The preprocessing workflow includes:

- Handling missing values using median imputation for numerical features.
- Handling missing values using most-frequent imputation for categorical features.
- One-hot encoding categorical variables.
- Standardizing numerical features where required by the algorithm.
- Removing invalid domain records and duplicate observations.

Preprocessing is implemented using Scikit-learn pipelines and a `ColumnTransformer`.

## Project Workflow

The project follows this general workflow:

1. Data loading and exploratory data analysis.
2. Data cleaning and preprocessing.
3. Train, validation, and test splitting.
4. Model training and comparison.
5. Hyperparameter optimization using Optuna.
6. Model selection using cross-validation average precision.
7. Classification threshold optimization using validation data.
8. Final model evaluation.
9. SHAP-based model interpretation.
10. Error analysis.
11. Model serialization.
12. FastAPI application development.
13. Frontend development.
14. GitHub version control and Render deployment.

## Model Development and Comparison

Seven classification algorithms were implemented and evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Extra Trees
5. HistGradientBoosting
6. XGBoost
7. CatBoost

Because loan default prediction involves class imbalance, Average Precision (AP), which summarizes the precision-recall curve, was selected as the primary model-selection metric.

### Model comparison results

The following table summarizes the cross-validation and test-set results obtained during development.

| Model | CV Average Precision | Test Average Precision | Test ROC-AUC |
|---|---:|---:|---:|
| Logistic Regression | 0.7309 | 0.7187 | 0.8702 |
| Decision Tree | 0.8556 | 0.8660 | 0.9176 |
| Random Forest | 0.8884 | 0.9017 | 0.9432 |
| Extra Trees | 0.8742 | 0.8813 | 0.9286 |
| HistGradientBoosting | 0.8979 | 0.9069 | 0.9480 |
| XGBoost | 0.9014 | 0.9127 | 0.9529 |
| CatBoost | **0.9022** | 0.9092 | 0.9492 |

CatBoost achieved the highest cross-validation Average Precision and was selected as the final model.

XGBoost was a close competitor.

## Hyperparameter Optimization

Optuna was used to optimize the hyperparameters of all seven classification models.

The optimization process used:

- Tree-structured Parzen Estimator (TPE) sampler.
- 20 trials per model.
- Stratified 3-fold cross-validation.
- Mean Average Precision as the optimization objective.

The objective was to identify hyperparameters that improved the model's ability to distinguish defaulting borrowers from non-defaulting borrowers.

CatBoost achieved the highest cross-validation Average Precision of **0.9022**.

## Model Evaluation

After selecting CatBoost, the classification threshold was optimized using the validation dataset.

The default classification threshold of 0.5 was not assumed to be optimal.

The selected threshold was approximately:

**0.2587**

This threshold was chosen by maximizing the F1-score on validation data.

### Final model results

| Metric | Result |
|---|---:|
| Accuracy | 0.9292 |
| Precision | 0.8652 |
| Recall | 0.8011 |
| F1-score | 0.8319 |
| ROC-AUC | 0.9492 |
| Average Precision | 0.9092 |
| Brier Score | 0.0495 |

### Confusion Matrix

| | Predicted Non-default | Predicted Default |
|---|---:|---:|
| Actual Non-default | 4,887 | 177 |
| Actual Default | 282 | 1,136 |

The model's predicted probabilities were also examined using calibration analysis. A sigmoid calibration experiment did not produce a meaningful improvement, so the original model probabilities were retained.

**Evaluation note:** The test set was consulted during model comparison and development. Consequently, these test metrics should be interpreted as development-stage evaluation results rather than an independent, untouched final benchmark.

## Model Explainability with SHAP

SHAP (SHapley Additive exPlanations) was used to understand how individual features influence the model's predictions.

Three visualizations were used:

### 1. SHAP Feature Importance

Identifies the features with the greatest overall influence on model predictions.

### 2. SHAP Beeswarm Plot

Shows the distribution of feature contributions across observations and illustrates how different feature values influence predictions.

### 3. SHAP Waterfall Plot

Explains how individual feature contributions move a specific prediction away from the model's baseline output.

Some of the influential features identified during analysis included:

- Loan-to-income ratio.
- Annual income.
- Home ownership.
- Interest rate.
- Loan grade.

These represent patterns learned by the model and should not be interpreted as causal relationships.

## Error Analysis

False positives and false negatives were examined to understand where the model makes incorrect predictions.

- **False Positive:** A non-defaulting borrower incorrectly classified as a defaulter.
- **False Negative:** A defaulter incorrectly classified as a non-defaulter.

The analysis included comparing numerical feature distributions and categorical feature frequencies across error groups.

This helped identify patterns in the model's mistakes and provided additional insight into its limitations.

## Application Architecture

The trained model is integrated into a FastAPI backend and served through an interactive web interface.

The application follows this architecture:

```text
                User
                  |
                  v
          HTML / CSS / JavaScript
                  |
                  v
              FastAPI
                  |
                  v
          Pydantic Validation
                  |
                  v
        Input DataFrame Creation
                  |
                  v
          Saved CatBoost Pipeline
                  |
                  v
          Default Probability
                  |
                  v
        Classification Threshold
                  |
                  v
           Prediction Response
                  |
                  v
          Frontend Result Display
```

### Backend

FastAPI provides the prediction endpoint.

Pydantic validates incoming loan application data, while the saved Scikit-learn pipeline handles preprocessing and prediction.

The model and its classification threshold are loaded during application startup.

### Frontend

The frontend is developed using:

- HTML for application structure.
- CSS for styling and animations.
- JavaScript for form interactions and API communication.

Users can enter borrower and loan information, submit the application, and view the resulting prediction.

### Prediction output

The API returns:

- Received input features.
- Predicted default probability.
- Classification threshold.
- Binary prediction.
- Prediction label.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Gradient Boosting | XGBoost, CatBoost |
| Hyperparameter Optimization | Optuna |
| Explainability | SHAP |
| API Development | FastAPI, Pydantic |
| Frontend | HTML, CSS, JavaScript |
| Model Serialization | Joblib |
| Environment and Dependency Management | uv |
| Version Control | Git, GitHub |
| Deployment | Render |

## Project Structure

```text
credit-risk-assessment/
│
├── artifacts/
│   ├── credit_risk_catboost.joblib
│   └── model_metadata.json
│
├── src/
│   └── credit_risk/
│       └── __init__.py
│
├── static/
│   ├── index.html
│   ├── styles.css
│   └── app.js
│
├── Credit Risk.ipynb
├── main.py
├── requirements.txt
├── pyproject.toml
├── uv.lock
├── render.yaml
├── .python-version
├── .gitignore
└── README.md
```

**Note:** The raw dataset and reference materials are not included in the repository.

## Run Locally

### Prerequisites

- Python 3.12
- pip or uv

### 1. Clone the repository

```bash
git clone https://github.com/Adithya-DSAIML/credit-risk-assessment.git
cd credit-risk-assessment
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the FastAPI application

```bash
uvicorn main:app --reload
```

### 5. Access the application

Frontend:

```text
http://127.0.0.1:8000/
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

## API Usage

The application exposes a prediction endpoint:

```text
POST /predict
```

### Example request

```json
{
  "person_age": 25,
  "person_income": 44004,
  "person_home_ownership": "RENT",
  "person_emp_length": 9,
  "loan_intent": "MEDICAL",
  "loan_grade": "E",
  "loan_amnt": 12000,
  "loan_int_rate": 16.45,
  "loan_percent_income": 0.27,
  "cb_person_default_on_file": "Y",
  "cb_person_cred_hist_length": 4
}
```

### Example response structure

```json
{
  "received_features": [],
  "default_probability": 0.9645,
  "classification_threshold": 0.2587,
  "prediction": 1,
  "prediction_label": "Default"
}
```

The probability shown above is illustrative of the example application used during development.

## Deployment

The application is deployed on Render.

The repository includes a `render.yaml` configuration file specifying the deployment configuration.

The deployed service uses:

**Build command**

```bash
pip install -r requirements.txt
```

**Start command**

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Access the deployed application:

[https://credit-risk-assessment-fix1.onrender.com](https://credit-risk-assessment-fix1.onrender.com)

## Limitations and Disclaimer

This project is intended for educational and portfolio demonstration purposes.

- Model predictions are based on patterns in the training dataset and may not generalize to every borrower population.
- Model performance metrics are based on the development dataset and should not be treated as independent production validation.
- SHAP explanations describe model behavior, not causal relationships.
- The predicted probability is not a guarantee of future repayment or default.
- The application is not intended to make real-world lending, credit approval, or financial decisions.

A production credit risk system would require additional validation, fairness and bias assessment, regulatory compliance, monitoring, security controls, and independent evaluation.

---

## Author

**Adithya Anand**

GitHub: [Adithya-DSAIML](https://github.com/Adithya-DSAIML)

**Project:** Credit Risk Assessment

**Live Demo:** [Credit Risk Assessment Application](https://credit-risk-assessment-fix1.onrender.com)
