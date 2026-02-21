# // import express from "express"
# // import bootstrap from "./src/app.controller.js";

# // const app = express();

# // bootstrap(app,express);

from fastapi import FastAPI
from pydantic import BaseModel  # New Import
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

# 1. Define a schema for the incoming data
class TextRequest(BaseModel):
    text: str

MODEL_NAME = "abhi099k/ai-text-detector-L0"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

@app.post("/predict")
# 2. Use the schema as the function argument
def predict(request: TextRequest):
    # Access the text via request.text
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probs, dim=-1).item()

    return {
        "prediction": "AI-Generated" if prediction == 1 else "Human-Written",
        "confidence": round(float(probs[0][prediction]), 4)
    }