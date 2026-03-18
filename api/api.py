from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI(title="OmicsAI API")

# load model
model = pickle.load(open("multiomics_model.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

@app.get("/")
def home():
    return {"message": "OmicsAI API running"}

@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    # handle missing features
    for col in features:
        if col not in df.columns:
            df[col] = 0

    df = df[features]

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }