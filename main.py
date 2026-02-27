# // import express from "express"
# // import bootstrap from "./src/app.controller.js";

# // const app = express();

# // bootstrap(app,express);

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel  # New Import
from transformers import AutoTokenizer, AutoModelForSequenceClassification, ViTImageProcessor, ViTForImageClassification
import torch
from PIL import Image
import io



app = FastAPI()


#Text Detection Model

# 1. Define a schema for the incoming data
class TextRequest(BaseModel):
    text: str

MODEL_NAME = "abhi099k/ai-text-detector-L0"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model_txt = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

@app.post("/predict")
# 2. Use the schema as the function argument
def predict(request: TextRequest):
    # Access the text via request.text
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model_txt(**inputs)
    
    probs = torch.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probs, dim=-1).item()

    return {
        "prediction": "AI-Generated" if prediction == 1 else "Human-Written",
        "confidence": round(float(probs[0][prediction]), 4)
    }


#Image Detection Model


MODEL_PATH = "./models/vit-deepfake/vit-deepfake-detection-model-v1-20251220_213720/model"

processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
model_img = ViTForImageClassification.from_pretrained(MODEL_PATH)

@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    # 1. Load image
    image = Image.open(io.BytesIO(await file.read()))
    
    # This removes the alpha channel from PNGs and ensures 3 channels (RGB)
    image = image.convert("RGB")
    
    # 2. Process for ViT
    inputs = processor(images=image, return_tensors="pt")
    
    # 3. Predict
    with torch.no_grad():
        outputs = model_img(**inputs)
    
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    
    # Assuming label 1 is 'Fake' based on standard deepfake models
    return {"label": "Fake" if prediction == 0 else "Real"}