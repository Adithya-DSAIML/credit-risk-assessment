from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Annotated, Optional, Literal, Any
import pandas as pd
import joblib
from pathlib import Path
from contextlib import asynccontextmanager
import json

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / 'static'

Age = Annotated[int, Field(..., ge=18, lt=100)]
Income = Annotated[float, Field(..., gt=0)]
HomeOwnership = Literal['RENT', 'OWN', 'MORTGAGE', "OTHER"]
EmploymentLength = Annotated[float, Field(ge=0)]
LoanIntent = Literal['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE',
                     'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION']
LoanGrade = Literal["A", "B", "C", "D", "E", "F", "G"]
LoanAmount = Annotated[float, Field(..., gt=0)]
InterestRate = Annotated[float, Field(...,gt=0)]
LoanPercentIncome = Annotated[float, Field(...,gt=0)]
PreviousDefault = Literal["Y", "N"]
CreditHistoryLength = Annotated[int, Field(..., ge=0)]

class LoanApplication(BaseModel):
    person_age: Age
    person_income: Income
    person_home_ownership : HomeOwnership
    person_emp_length : Optional[EmploymentLength] = None
    loan_intent : LoanIntent
    loan_grade : LoanGrade
    loan_amnt : LoanAmount
    loan_int_rate : InterestRate
    loan_percent_income : LoanPercentIncome
    cb_person_default_on_file : PreviousDefault
    cb_person_cred_hist_length : CreditHistoryLength

class PredictionResponse(BaseModel):
    received_features : list[dict[str, Any]]
    default_probability : float
    classification_threshold: float
    prediction: Literal[0, 1]
    prediction_label: Literal['Default', 'Non-default']


ml_model = {}
@asynccontextmanager
async def lifespan(app : FastAPI):
    model_path = (Path(__file__).resolve().parent / 'artifacts' / 'credit_risk_catboost.joblib')
    metadata_path = (Path(__file__).resolve().parent / 'artifacts' / 'model_metadata.json')

    ml_model['model'] = joblib.load(model_path)

    with open(metadata_path, 'r') as file:
        metadata = json.load(file)
    ml_model['threshold'] = metadata['classification_threshold']
    ml_model['model_name'] = metadata['model_name']

    print('Model loaded successfully!')

    yield

    ml_model.clear()

app = FastAPI(description='Predict Default Risk',
              version='1.0', 
              title='Credit Risk Assessment API',
              lifespan=lifespan)

@app.get('/', include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / 'index.html')

@app.post('/predict', response_model=PredictionResponse)
def predict(data: LoanApplication):
    input_df = pd.DataFrame([data.model_dump()])
    default_probability = ml_model['model'].predict_proba(input_df)[0,1]
    threshold = ml_model['threshold']
    prediction = int(default_probability >= threshold)
    return {
        'received_features' : input_df.to_dict(orient='records'),
        'default_probability' : float(default_probability),
        'classification_threshold' : float(threshold),
        'prediction' : prediction,
        'prediction_label' : 'Default' if prediction == 1 else 'Non-default'
        }


app.mount('/static', StaticFiles(directory=STATIC_DIR, html=True), name='static')