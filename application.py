from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_name = "XGBoost.pkl"
model = joblib.load(model_name)

class UserInput(BaseModel):
    gender: str
    age: float
    height: float
    weight: float
    duration: float
    heart: float
    body: float

@app.post("/predict")
def predict_calories(input: UserInput):
    gender_val = 1 if input.gender.lower() == "male" else 0

    df = pd.DataFrame([{
        "Gender": gender_val,
        "Age": input.age,
        "Height": input.height,
        "Weight": input.weight,
        "Duration": input.duration,
        "Heart_Rate": input.heart,
        "Body_Temp": input.body
    }])

    prediction = model.predict(df)

    return {"predicted_calories": round(float(prediction[0]), 2)}
