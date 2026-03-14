from fastapi import FastAPI
import pickle
import uvicorn

# Create FastAPI app
app = FastAPI(title="AI Text Detection API")

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Home route
@app.get("/")
def home():
    return {
        "message": "AI Text Detection API is running"
    }


# Prediction route
@app.get("/predict")
def predict(text: str):

    # Convert text into TF-IDF vector
    vector = vectorizer.transform([text])

    # Make prediction
    prediction = model.predict(vector)[0]

    # Convert result to readable output
    if prediction == 1:
        result = "AI Generated"
    else:
        result = "Human Written"

    return {
        "input_text": text,
        "prediction": result
    }


# Run server locally
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
