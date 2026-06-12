import os
import io
import json
import string
from http.client import HTTPException
from typing import Dict, List
from agent import ClaimRequest, MisinformationDetectionAgent, VerificationResponse
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, ViTImageProcessor, ViTForImageClassification
import torch
import numpy as np
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI  # Re-activated client imports
import os
from dotenv import load_dotenv

# Automatically finds and parses your root .env file
load_dotenv()

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

# OpenAI Client Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ------------------------------------------------------------------
# 2. SCHEMAS & PROMPTS
# ------------------------------------------------------------------
class TextRequest(BaseModel):
    text: str


# 1. Create a clear schema for an individual token element block
class TokenAttribution(BaseModel):
    token: str
    score: float

# 2. Swap out List[Dict] with your explicit new schema class object type
class TokenExplanationRequest(BaseModel):
    text: str
    prediction: str
    confidence: float
    tokens: List[TokenAttribution]

SYSTEM_PROMPT = """You are the advanced linguistic forensic engine for "HATS" (Human & AI Truth Scanner), a real-time browser extension that evaluates social media posts. Your job is to read an analysis payload from our model and write a sharp, highly engaging explanation (strictly 1-2 sentences) explaining the verdict in a casual, natural, social-media-friendly tone.

### Guidelines for Generating Insights:
1. Contextual Awareness: Look at the meaning and topic of the input text. If it is a casual Facebook rant, a tech tweet, or a corporate LinkedIn job post, tailor your commentary to that environment.
2. Burstiness & Predictability: Focus on the rhythm of the writing. Note that AI writes with mechanical consistency (low burstiness) and overly sanitized, predictable phrasing (low perplexity). Human text features organic irregularities, erratic rhythms, slang, or emotional beats.
3. Don't Just List Tokens: Do not say "Token X has a score of Y." Instead, weave the top-scoring tokens conceptually into your critique of the text's tone or structural style.

### Execution Rules:
- Length: Strictly 1 to 2 sentences. Keep it punchy so it fits comfortably inside a small browser popup or tooltip wrapper.
- Tone: Natural, observant, modern, and social-media-savvy. Never use rigid AI filler phrases like "Based on the provided payload" or "The data indicates."
- Formatting: Return ONLY a valid JSON object with a single key "explanation". Do not wrap the JSON in markdown triple-backticks or code blocks."""

# ------------------------------------------------------------------
# 3. HELPER FUNCTIONS (Integrated Gradients)
# ------------------------------------------------------------------
def run_integrated_gradients(text: str, pred_class: int, steps: int = 2, internal_batch_size: int = 2):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    embedding_layer = model_txt.get_input_embeddings()
    input_embeddings = embedding_layer(input_ids)

    model_dtype = next(model_txt.parameters()).dtype
    input_embeddings = input_embeddings.to(model_dtype)

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    baseline_ids = torch.full_like(input_ids, pad_token_id)
    baseline_embeddings = embedding_layer(baseline_ids).to(model_dtype)

    step_sizes = torch.linspace(0, 1, steps, device=device, dtype=model_dtype)
    scaled_inputs_all = baseline_embeddings + step_sizes.view(-1, 1, 1) * (input_embeddings - baseline_embeddings)

    all_grads = []
    for i in range(0, steps, internal_batch_size):
        chunk_scaled = scaled_inputs_all[i:i + internal_batch_size].detach().requires_grad_(True)
        actual_chunk_size = chunk_scaled.shape[0]
        chunk_mask = attention_mask.repeat(actual_chunk_size, 1)

        if device == "cuda":
            with torch.amp.autocast("cuda"):
                outputs = model_txt(inputs_embeds=chunk_scaled, attention_mask=chunk_mask)
                target_logits = outputs.logits[:, pred_class]
            grad = torch.autograd.grad(outputs=target_logits, inputs=chunk_scaled, grad_outputs=torch.ones_like(target_logits))[0]
        else:
            outputs = model_txt(inputs_embeds=chunk_scaled, attention_mask=chunk_mask)
            target_logits = outputs.logits[:, pred_class]
            grad = torch.autograd.grad(outputs=target_logits, inputs=chunk_scaled, grad_outputs=torch.ones_like(target_logits))[0]

        all_grads.append(grad.detach())

    total_grads = torch.cat(all_grads, dim=0)
    avg_grads = total_grads.mean(dim=0)

    attributions = (input_embeddings.squeeze(0) - baseline_embeddings.squeeze(0)) * avg_grads
    token_importance = attributions.sum(dim=-1)

    scores = token_importance.detach().float().cpu().numpy()
    scores = scores / (np.max(np.abs(scores)) + 1e-8)

    raw_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    interpreted_results = []

    for token, score in zip(raw_tokens[1:-1], scores[1:-1]):
        clean_token = token.replace("Ġ", "").replace("▁", "").strip()
        if not clean_token or token.startswith("["):
            continue
        interpreted_results.append({
            "token": token,  # Send the tokenizer string intact so frontend parses spacing symbols correctly
            "score": round(float(score), 4)
        })

    return interpreted_results

# ------------------------------------------------------------------
# 4. API ENDPOINTS (Pipeline Partition)
# ------------------------------------------------------------------

@app.post("/predict")
def predict(request: TextRequest):
    """Pipeline Stage 1: Fast Classification Inference Only (<150ms)."""
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        if device == "cuda":
            with torch.amp.autocast('cuda'):
                outputs = model_txt(**inputs)
        else:
            outputs = model_txt(**inputs)
    
    probs = torch.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probs, dim=-1).item()
    
    label = "AI-Generated" if prediction == 1 else "Human-Written"
    confidence = round(float(probs[0][prediction]), 4)

    return {
        "prediction": label,
        "confidence": confidence
    }

@app.post("/explain-tokens")
def explain_tokens(request: TextRequest):
    """Pipeline Stage 2: Executes background XAI Integrated Gradients if text is AI."""
    # Compute relative targeting specifically towards the AI-Generated feature node (Class 1)
    token_attributions = run_integrated_gradients(request.text, pred_class=1, steps=2, internal_batch_size=2)
    return {"tokens": token_attributions}

@app.post("/explain-linguistics")
def explain_linguistics(payload: TokenExplanationRequest):
    """Pipeline Stage 3: Asynchronously called by extension to build user-friendly insights."""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": payload.json()}
            ],
            response_format={"type": "json_object"},
            temperature=0.5
        )
        llm_output = json.loads(response.choices[0].message.content)
        explanation = llm_output.get("explanation", "Linguistic patterns verified.")
    except Exception as e:
        explanation = f"Linguistic insight processing engine unavailable. ({str(e)})"
        
    return {"explanation": explanation}

# ------------------------------------------------------------------
# EXISTING IMAGE & FACT CHECKING ENDPOINTS (Unchanged)
# ------------------------------------------------------------------
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