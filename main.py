import os
import io
import json
from fastapi import HTTPException
from typing import Dict, List
import string
from agent import ClaimRequest, MisinformationDetectionAgent, VerificationResponse
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, ViTImageProcessor, ViTForImageClassification
import torch
import numpy as np
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
# from openai import OpenAI  # Commented out GPT-4o-mini client import

app = FastAPI()

# Allow all origins during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# 1. INITIALIZE CLIENTS & MODELS
# ------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# Text Detection Model
MODEL_NAME = "./models/text-detector"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Optimize memory matching your requirements (.half() if on GPU)
if device == "cuda":
    model_txt = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).half().to(device)
else:
    model_txt = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
model_txt.eval()

# Image Detection Model
MODEL_PATH = "./models/vit-deepfake/vit-deepfake-detection-model-v1-20251220_213720/model"
processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
model_img = ViTForImageClassification.from_pretrained(MODEL_PATH).to(device)
model_img.eval()

# Fact-Checking Agent Keys
GROQ_API_KEY   = os.getenv("GROQ_API_KEY",   "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")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "f4c8d4545449d1ec47180589209e220bf70bcc4a")
agent = MisinformationDetectionAgent(GROQ_API_KEY, SERPER_API_KEY)

# OpenAI Client Configuration (Commented out for now)
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ------------------------------------------------------------------
# 2. SCHEMAS & PROMPTS (Prompt Commented Out)
# ------------------------------------------------------------------
class TextRequest(BaseModel):
    text: str

# SYSTEM_PROMPT = """You are the explanation engine for a browser extension..."""

# ------------------------------------------------------------------
# 3. HELPER FUNCTIONS (Integrated Gradients)
# ------------------------------------------------------------------

def run_integrated_gradients(
    text: str,
    pred_class: int,
    steps: int = 2,
    internal_batch_size: int = 2
):
    """
    Optimized Integrated Gradients.

    Improvements:
    - No GPU -> CPU transfers inside loop
    - Keeps all gradients on GPU
    - Handles FP16 models correctly
    - Cleans SentencePiece / BPE tokens
    - Lower latency
    """

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    embedding_layer = model_txt.get_input_embeddings()

    input_embeddings = embedding_layer(input_ids)

    model_dtype = next(model_txt.parameters()).dtype
    input_embeddings = input_embeddings.to(model_dtype)

    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else 0
    )

    baseline_ids = torch.full_like(
        input_ids,
        pad_token_id
    )

    baseline_embeddings = embedding_layer(
        baseline_ids
    ).to(model_dtype)

    step_sizes = torch.linspace(
        0,
        1,
        steps,
        device=device,
        dtype=model_dtype
    )

    scaled_inputs_all = (
        baseline_embeddings
        + step_sizes.view(-1, 1, 1)
        * (input_embeddings - baseline_embeddings)
    )

    all_grads = []

    for i in range(0, steps, internal_batch_size):

        chunk_scaled = (
            scaled_inputs_all[i:i + internal_batch_size]
            .detach()
            .requires_grad_(True)
        )

        actual_chunk_size = chunk_scaled.shape[0]

        chunk_mask = attention_mask.repeat(
            actual_chunk_size,
            1
        )

        if device == "cuda":

            with torch.amp.autocast("cuda"):

                outputs = model_txt(
                    inputs_embeds=chunk_scaled,
                    attention_mask=chunk_mask
                )

                target_logits = outputs.logits[
                    :,
                    pred_class
                ]

            grad = torch.autograd.grad(
                outputs=target_logits,
                inputs=chunk_scaled,
                grad_outputs=torch.ones_like(
                    target_logits
                )
            )[0]

        else:

            outputs = model_txt(
                inputs_embeds=chunk_scaled,
                attention_mask=chunk_mask
            )

            target_logits = outputs.logits[
                :,
                pred_class
            ]

            grad = torch.autograd.grad(
                outputs=target_logits,
                inputs=chunk_scaled,
                grad_outputs=torch.ones_like(
                    target_logits
                )
            )[0]

        # KEEP ON GPU
        all_grads.append(grad.detach())

    total_grads = torch.cat(
        all_grads,
        dim=0
    )

    avg_grads = total_grads.mean(dim=0)

    attributions = (
        input_embeddings.squeeze(0)
        - baseline_embeddings.squeeze(0)
    ) * avg_grads

    token_importance = attributions.sum(dim=-1)

    scores = (
        token_importance
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    scores = scores / (
        np.max(np.abs(scores))
        + 1e-8
    )

    raw_tokens = tokenizer.convert_ids_to_tokens(
        input_ids[0]
    )

    punctuation_set = set(string.punctuation)

    interpreted_results = []

    for token, score in zip(
        raw_tokens[1:-1],
        scores[1:-1]
    ):

        clean_token = (
            token.replace("Ġ", "")
                 .replace("▁", "")
                 .strip()
        )

        if not clean_token or token.startswith("["):
            continue

        interpreted_results.append({
            "token": clean_token,
            "score": round(float(score), 4)
        })

    # interpreted_results.sort(
    #     key=lambda x: x["score"],
    #     reverse=True
    # )

    return interpreted_results
# ------------------------------------------------------------------
# 4. API ENDPOINTS
# ------------------------------------------------------------------

@app.post("/predict")
def predict(request: TextRequest):
    # 1. Standard Prediction Inference
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        if device == "cuda":
            with torch.amp.autocast('cuda'):
                outputs = model_txt(**inputs)
        else:
            outputs = model_txt(**inputs)
    
    probs = torch.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probs, dim=-1).item()
    
    is_ai = (prediction == 1)
    label = "AI-Generated" if is_ai else "Human-Written"
    confidence = round(float(probs[0][prediction]), 4)

    token_attributions = []

    # 2. Run IG ONLY if the content is classified as AI
    if is_ai:
        try:
            token_attributions = run_integrated_gradients(request.text,pred_class=1,steps=2,internal_batch_size=2)
        except Exception as e:
            print(f"XAI Processing Error: {e}")

    # 3. Fetch user-friendly explanation via GPT-4o-mini (Commented out for now)
    # try:
    #     llm_payload = {"text": request.text, "classification": label}
    #     if is_ai:
    #         top_tokens = [t for t in token_attributions if t["score"] > 0.1]
    #         llm_payload["top_influential_tokens"] = sorted(top_tokens, key=lambda x: x["score"], reverse=True)[:5]
    #
    #     response = openai_client.chat.completions.create(
    #         model="gpt-4o-mini",
    #         messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": json.dumps(llm_payload)}],
    #         response_format={"type": "json_object"},
    #         temperature=0.3
    #     )
    #     llm_output = json.loads(response.choices[0].message.content)
    #     explanation = llm_output.get("explanation", "Analysis concluded successfully.")
    # except Exception as e:
    #     explanation = f"Could not generate automated description: {str(e)}"

    # 4. Return unified analytical payload back to browser extension
    return {
        "prediction": label,
        "confidence": confidence,
        # "explanation": explanation,  # Commented out for now
        "tokens": token_attributions if is_ai else []
    }


@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model_img(**inputs)
    
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    return {"label": "Fake" if prediction == 0 else "Real"}


@app.get("/", tags=["Health"])
def root():
    return {
        "status": "ok",
        "service": "Misinformation & Deepfake Detection API",
        "model": agent.model_name,
        "date": agent.get_current_date_info()["full_date"],
    }


@app.post("/verify", response_model=VerificationResponse, tags=["Fact-checking"])
def verify_claim(body: ClaimRequest):
    try:
        result = agent.detect_misinformation(body.claim, prioritize_news=body.prioritize_news)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
 
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
 
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
        search_results=normalised_sources,
    )

