# // import express from "express"
# // import bootstrap from "./src/app.controller.js";

# // const app = express();

# // bootstrap(app,express);

from ast import List
from http.client import HTTPException
from typing import Dict

from agent import ClaimRequest, MisinformationDetectionAgent, VerificationResponse
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel  # New Import
from transformers import AutoTokenizer, AutoModelForSequenceClassification, ViTImageProcessor, ViTForImageClassification
import torch
from PIL import Image
import io
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()

# Allow all origins during development (you can restrict later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],                # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],                # Allows all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],                # Allows all headers
)


#Text Detection Model

# 1. Define a schema for the incoming data
class TextRequest(BaseModel):
    text: str

MODEL_NAME = "./models/text-detector"
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


import os 
GROQ_API_KEY   = os.getenv("GROQ_API_KEY",   "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")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "ee285149e0f2ce50f74d1dae254489bbce93b141")
SELECTED_MODEL = os.getenv("GROQ_MODEL",     "openai/gpt-oss-120b")

# Singleton agent instance (reuses the same HTTP session across requests)
agent = MisinformationDetectionAgent(GROQ_API_KEY,SERPER_API_KEY)
 
 
@app.get("/", tags=["Health"])
def root():
    """Health-check endpoint."""
    return {
        "status": "ok",
        "service": "Misinformation Detection API",
        "model": agent.model_name,
        "date": agent.get_current_date_info()["full_date"],
    }
 
 
@app.post("/verify", response_model=VerificationResponse, tags=["Fact-checking"])
def verify_claim(body: ClaimRequest):
    """
    Verify a claim and return a detailed fact-check report.
 
    - **claim**: The statement you want to fact-check (min 5 characters).
    - **prioritize_news**: `true` (default) uses news from the last 3 days;
      `false` uses the last 7 days.
 
    Returns a structured report including:
    - LLM verdict (TRUE / FALSE / PARTIALLY TRUE / MISLEADING / UNVERIFIABLE / OUTDATED)
    - Confidence and timeliness fields
    - Ranked sources with per-source credibility scores
    """
    try:
        result = agent.detect_misinformation(body.claim, prioritize_news=body.prioritize_news)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
 
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
 
    # Normalise search_results so Pydantic can validate them
    normalised_sources: List[Dict] = []
    for src in result.get("search_results", []):
        cred = src.get("credibility", {})
        normalised_sources.append({
            "title":       src.get("title", ""),
            "snippet":     src.get("snippet", ""),
            "link":        src.get("link", ""),
            "source_type": src.get("source_type", "web"),
            "date":        src.get("date"),
            "source":      src.get("source"),
            "credibility": {
                "url":               cred.get("url", src.get("link", "")),
                "domain":            cred.get("domain", ""),
                "credibility_score": cred.get("credibility_score", 0.0),
                "credibility_level": cred.get("credibility_level", "UNKNOWN"),
                "source_type":       cred.get("source_type", "unknown"),
                "reasoning":         cred.get("reasoning", []),
                "is_trusted":        cred.get("is_trusted", False),
            },
        })
 
    return VerificationResponse(
        claim=result["claim"],
        analysis=result["analysis"],
        analysis_date=result.get("analysis_date"),
        aggregate_credibility=result.get("aggregate_credibility"),
        high_cred_sources_count=result.get("high_cred_sources_count"),
        search_results=normalised_sources,  # type: ignore[arg-type]
    )