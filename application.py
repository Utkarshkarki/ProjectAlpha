from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.pipeline.predict_pipeline import PredictPipeline

app = FastAPI(title="ProjectAlpha Income Prediction API")

# Load model and preprocessor once at startup
predictor = PredictPipeline()

class IncomeFeatures(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        fields = {
            "education_num": "education.num",
            "marital_status": "marital.status",
            "capital_gain": "capital.gain",
            "capital_loss": "capital.loss",
            "hours_per_week": "hours.per.week",
            "native_country": "native.country",
        }

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict_income(features: IncomeFeatures):
    try:
        input_dict = features.dict(by_alias=True)
        prediction = predictor.predict(input_dict)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
