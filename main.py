from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import os

app = FastAPI()

# Path setup
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "train.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")
# Load model
with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

# Input format
class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "API running 🚀"}

@app.post("/predict")
def predict(input: TextInput):
    vec = vectorizer.transform([input.text])

    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]

    return {
        "prediction": "AI Generated Text" if pred == 1 else "Human Written Text",
        "confidence": float(max(prob))
    }
