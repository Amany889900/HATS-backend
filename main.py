import os
import io
import json
import math
import uuid
import base64
import string
import warnings
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import open_clip
from PIL import Image, ImageDraw

from agent import ClaimRequest, MisinformationDetectionAgent, VerificationResponse
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    ViTImageProcessor,
    ViTForImageClassification,
)
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

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

# --- Text Detection Model ---
MODEL_NAME = "./models/text-detector"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if device == "cuda":
    model_txt = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).half().to(device)
else:
    model_txt = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
model_txt.eval()

# --- Image Detection Model (fast predict-image endpoint) ---
MODEL_PATH = "./models/vit-deepfake/vit-deepfake-detection-model-v1-20251220_213720/model"
processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
model_img = ViTForImageClassification.from_pretrained(MODEL_PATH).to(device)
model_img.eval()

# --- Deepfake Explainer Model (output_attentions=True for rollout) ---
print("Loading deepfake explainer model …")
explainer_processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
explainer_model = ViTForImageClassification.from_pretrained(MODEL_PATH, output_attentions=True).to(device)
explainer_model.eval()
print("✅ Deepfake explainer model loaded")

# --- CLIP Model ---
print("Loading CLIP model (ViT-B/32) …")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
clip_model = clip_model.to(device).eval()
clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
print("✅ CLIP loaded")

# --- Fact-Checking Agent ---
GROQ_API_KEY   = os.getenv("GROQ_API_KEY",   "")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
agent = MisinformationDetectionAgent(GROQ_API_KEY, SERPER_API_KEY)

# --- OpenAI Client (lazy — only created when first used so a missing key won't crash startup) ---
_openai_client: OpenAI | None = None

def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set. Add it to your .env file.")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client

# ------------------------------------------------------------------
# 2. SCHEMAS & PROMPTS
# ------------------------------------------------------------------
class TextRequest(BaseModel):
    text: str


class TokenAttribution(BaseModel):
    token: str
    score: float


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
# 3. CLIP CONTENT CLASSIFICATION DATA
# ------------------------------------------------------------------
CLIP_CANDIDATES = [
    ("a close-up portrait photo of a single human face",              "face"),
    ("a photo of a group or crowd of people at an event",             "crowd"),
    ("a selfie photo taken by a person holding a camera",             "selfie"),
    ("a photo of a couple or two people posing together",             "couple"),
    ("a photo of nature, forests, mountains, sky or water",           "nature"),
    ("a photo of a cityscape, buildings, streets or urban scene",     "urban"),
    ("a photo of food, a meal, a drink or a dish on a table",         "food"),
    ("a photo of an animal or pet",                                   "animal"),
    ("a photo of a sports event, athlete or fitness activity",        "sports"),
    ("a screenshot, meme, graphic design or digital artwork",         "graphic"),
    ("a photo of a product, gadget, fashion item or merchandise",     "product"),
    ("a photo of a travel destination, landmark or tourist spot",     "travel"),
]
CLIP_CANDIDATE_TEXTS = [t for t, _ in CLIP_CANDIDATES]
CLIP_LABEL_MAP       = [l for _, l in CLIP_CANDIDATES]

GRID_LABELS = {
    "face":    [["forehead",    "forehead",     "forehead"],
                ["eyes",        "nose",         "eyes"],
                ["mouth",       "skin",         "jaw"]],
    "selfie":  [["hairline",    "forehead",     "hairline"],
                ["eyes",        "nose",         "ear"],
                ["mouth",       "jaw",          "neck"]],
    "couple":  [["background",  "background",   "background"],
                ["skin",        "eyes",         "skin"],
                ["jaw",         "mouth",        "jaw"]],
    "crowd":   [["background",  "background",   "background"],
                ["skin",        "nose",         "skin"],
                ["mouth",       "skin",         "mouth"]],
    "nature":  [["sky",         "sky",          "horizon"],
                ["foliage",     "texture",      "foliage"],
                ["water",       "edges",        "edges"]],
    "urban":   [["sky",         "architecture", "windows"],
                ["signs",       "architecture", "windows"],
                ["pavement",    "edges",        "pavement"]],
    "food":    [["background",  "lighting",     "background"],
                ["food_main",   "food_main",    "food_main"],
                ["plate",       "table",        "shadow"]],
    "animal":  [["background",  "background",   "background"],
                ["animal_face", "fur",          "animal_face"],
                ["paws",        "fur",          "paws"]],
    "sports":  [["background",  "background",   "background"],
                ["jersey",      "motion_blur",  "jersey"],
                ["equipment",   "edges",        "equipment"]],
    "graphic": [["text_overlay","logo",         "text_overlay"],
                ["texture",     "texture",      "texture"],
                ["border",      "text_overlay", "border"]],
    "product": [["background",  "lighting",     "background"],
                ["product_main","label",        "product_main"],
                ["shadow",      "edges",        "shadow"]],
    "travel":  [["sky",         "landmark",     "sky"],
                ["landmark",    "landmark",     "crowd_bg"],
                ["pathway",     "edges",        "pathway"]],
}

REGION_DESCRIPTIONS = {
    "eyes":         ("the eye area",         "Eyes are extremely difficult for AI to generate — small imperfections or mismatched reflections give it away."),
    "nose":         ("the nose",             "The nose bridge and tip often show blending artifacts where the AI stitched different parts together."),
    "mouth":        ("the mouth area",       "Lips and teeth are a common failure point — unnatural smoothness or slight misalignment often appears here."),
    "hairline":     ("the hairline",         "Hair edges are where AI generation breaks down most visibly — strands often blur or repeat unnaturally."),
    "jaw":          ("the jawline",          "The boundary between face and background is a common failure point for AI face generators."),
    "skin":         ("the skin texture",     "Real skin has natural variation. AI-generated skin tends to look uniformly smooth or 'plastic'."),
    "forehead":     ("the forehead",         "Foreheads can reveal subtle hairline cloning or skin-tone stitching artifacts common in GAN faces."),
    "ear":          ("the ear area",         "Ears are notoriously hard for AI — they often appear asymmetrical, melted or geometrically inconsistent."),
    "neck":         ("the neck/shoulder",    "The transition from face to neck/shoulders often shows seam-like blending artifacts in synthetic images."),
    "hand":         ("the hands",            "AI models consistently struggle with hands — extra fingers, fused digits or unnatural proportions are telltale signs."),
    "background":   ("the background",       "Backgrounds often show lighting or depth inconsistencies that wouldn't appear in a real photograph."),
    "lighting":     ("the lighting",         "The way light falls on the subject doesn't match the environment — a common sign of AI compositing."),
    "edges":        ("the object edges",     "Subject-background boundaries often show 'halos' or blending artifacts typical of generative models."),
    "texture":      ("the surface detail",   "The model found unnatural smoothness or repetitive patterns in the fine surface details."),
    "foliage":      ("leaves and plants",    "AI often struggles with complex overlapping leaf patterns, creating repetitive or blurry textures."),
    "sky":          ("the sky/clouds",       "Clouds in AI images can have unnatural swirls or lighting that doesn't match the rest of the scene."),
    "water":        ("the water surface",    "Reflections and ripples are hard for AI to simulate, often leading to geometric inconsistencies."),
    "horizon":      ("the horizon line",     "A misaligned or unnaturally sharp horizon is a subtle but reliable indicator of AI-generated scenery."),
    "windows":      ("the windows",          "Window reflections and glass surfaces are highly complex — AI often renders them with inconsistent reflections."),
    "signs":        ("signs and text",       "Text in AI images is almost always distorted, misspelled or contains nonsensical characters."),
    "pavement":     ("the pavement/ground",  "Ground textures (tiles, asphalt, cobblestones) often repeat or blur unnaturally in AI scenes."),
    "architecture": ("the building edges",   "Straight architectural lines often subtly curve or wobble in AI-generated urban imagery."),
    "food_main":    ("the main dish",        "AI food images often look 'too perfect' — unnaturally glossy surfaces or physically impossible shapes."),
    "plate":        ("the plate/bowl",       "Plate edges and rim patterns tend to be geometrically inconsistent or subtly broken in AI images."),
    "table":        ("the table surface",    "Table textures and reflections are a common failure point — surfaces often look artificially smooth."),
    "animal_face":  ("the animal's face",    "Animal eyes and snouts generated by AI often show subtle blurring or unnatural symmetry."),
    "fur":          ("the fur/feathers",     "Complex fur or feather textures tend to become repetitive or smeared in AI-generated animal images."),
    "paws":         ("the paws/claws",       "Extremities like paws are structurally complex — AI often merges or distorts their anatomy."),
    "jersey":       ("the jersey/uniform",   "Sports jerseys with numbers and logos are difficult for AI — text and logos are frequently distorted."),
    "motion_blur":  ("motion and blur",      "AI struggles to simulate realistic motion blur — movement often looks frozen or artificially smeared."),
    "equipment":    ("sports equipment",     "Balls, racquets, and other equipment often have subtle shape distortions in AI-generated sports images."),
    "text_overlay": ("text overlays",        "Text in AI-generated graphics is commonly garbled — letters merge, float or make no semantic sense."),
    "logo":         ("logos and icons",      "Brand logos and icons are frequently distorted or subtly wrong in AI-generated digital content."),
    "border":       ("image borders/frame",  "The outer frame or overlay elements often show inconsistent anti-aliasing typical of AI compositing."),
    "product_main": ("the product",          "Product surfaces in AI images often have unnaturally perfect reflections or impossible geometry."),
    "label":        ("the product label",    "Labels and branding on products are almost always distorted or contain unreadable text in AI images."),
    "shadow":       ("the shadow",           "Cast shadows are computationally hard — AI often places them in physically inconsistent directions."),
    "landmark":     ("the landmark",         "Famous structures generated by AI often have subtle structural errors or wrong proportions."),
    "crowd_bg":     ("the crowd in background", "Background people in travel shots are a common place for AI to introduce clone-like repetitions."),
    "pathway":      ("the pathway/road",     "Paths and roads often show perspective or tiling inconsistencies in AI travel imagery."),
}

ANALOGY_BANK = {
    "face":    "Think of it like a painting of a face vs a photograph. A skilled painting can look real, but up close brushstrokes appear where a camera would show fine detail. AI faces have their own version of 'brushstrokes'.",
    "selfie":  "Think of a fun-house mirror — it looks like you, but small proportions are slightly off. AI selfies have that same subtle wrongness in the ears, hairline and neck that your brain registers before your eyes do.",
    "couple":  "Think of a photo collage where two images were blended together. Real couple photos have consistent lighting on both people; AI often lights each person differently as if they came from separate photos.",
    "crowd":   "Think of copy-paste in a photo editor. AI struggles with groups — faces subtly repeat, blend into each other, or have lighting that doesn't match, as if each person came from a different photograph.",
    "nature":  "Think of a CGI nature documentary vs a real one. CGI looks stunning but has a 'too perfect' or 'too smooth' quality that real nature never has. The model looks for those same unnatural patterns.",
    "urban":   "Think of a CGI city in a movie vs a real street photo. AI urban scenes look polished but text on signs is garbled, building edges wobble and window reflections don't match the sky.",
    "food":    "Think of a menu photo vs a restaurant's actual dish. AI food looks impossibly perfect — unnaturally glossy, geometrically flawless — in a way that real food photographed under real lighting never does.",
    "animal":  "Think of a stuffed toy vs a real animal. AI animals often look slightly too symmetrical, with fur that repeats like wallpaper and eyes that lack the wet, uneven quality of a real creature.",
    "sports":  "Think of a video-game screenshot vs a real sports photo. AI sports imagery freezes motion unnaturally, puts jersey numbers in the wrong font and gives equipment impossible shapes.",
    "graphic": "Think of a professional designer vs an AI image generator trying to match their style. The layout looks plausible at a glance, but text is garbled, logos are subtly wrong and edges lack the precision of real design work.",
    "product": "Think of an official brand photo vs a counterfeit catalogue. AI product images have suspiciously perfect surfaces, impossible reflections and labels that look right from a distance but dissolve under scrutiny.",
    "travel":  "Think of a postcard vs a real travel photo. AI landmarks have subtly wrong proportions, background crowds contain repeated faces, and the paths and roads have a tiled, repetitive quality real scenes never have.",
}

# ------------------------------------------------------------------
# 4. HELPER FUNCTIONS — Text XAI (Integrated Gradients)
# ------------------------------------------------------------------
def run_integrated_gradients(text: str, pred_class: int, steps: int = 2, internal_batch_size: int = 2):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids      = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    embedding_layer  = model_txt.get_input_embeddings()
    input_embeddings = embedding_layer(input_ids)

    model_dtype      = next(model_txt.parameters()).dtype
    input_embeddings = input_embeddings.to(model_dtype)

    pad_token_id        = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    baseline_ids        = torch.full_like(input_ids, pad_token_id)
    baseline_embeddings = embedding_layer(baseline_ids).to(model_dtype)

    step_sizes        = torch.linspace(0, 1, steps, device=device, dtype=model_dtype)
    scaled_inputs_all = baseline_embeddings + step_sizes.view(-1, 1, 1) * (input_embeddings - baseline_embeddings)

    all_grads = []
    for i in range(0, steps, internal_batch_size):
        chunk_scaled      = scaled_inputs_all[i:i + internal_batch_size].detach().requires_grad_(True)
        actual_chunk_size = chunk_scaled.shape[0]
        chunk_mask        = attention_mask.repeat(actual_chunk_size, 1)

        if device == "cuda":
            with torch.amp.autocast("cuda"):
                outputs      = model_txt(inputs_embeds=chunk_scaled, attention_mask=chunk_mask)
                target_logits = outputs.logits[:, pred_class]
            grad = torch.autograd.grad(outputs=target_logits, inputs=chunk_scaled,
                                       grad_outputs=torch.ones_like(target_logits))[0]
        else:
            outputs       = model_txt(inputs_embeds=chunk_scaled, attention_mask=chunk_mask)
            target_logits = outputs.logits[:, pred_class]
            grad = torch.autograd.grad(outputs=target_logits, inputs=chunk_scaled,
                                       grad_outputs=torch.ones_like(target_logits))[0]

        all_grads.append(grad.detach())

    total_grads  = torch.cat(all_grads, dim=0)
    avg_grads    = total_grads.mean(dim=0)
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
            "token": token,
            "score": round(float(score), 4)
        })
    return interpreted_results


# ------------------------------------------------------------------
# 5. HELPER FUNCTIONS — Image XAI (CLIP + Attention Rollout)
# ------------------------------------------------------------------
def classify_image_type_clip(pil_image):
    img_tensor  = clip_preprocess(pil_image).unsqueeze(0).to(device)
    text_tokens = clip_tokenizer(CLIP_CANDIDATE_TEXTS).to(device)
    with torch.no_grad():
        img_feat = clip_model.encode_image(img_tensor)
        txt_feat = clip_model.encode_text(text_tokens)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        logits   = (100.0 * img_feat @ txt_feat.T).softmax(dim=-1)
    probs      = logits[0].cpu().numpy()
    best_idx   = int(probs.argmax())
    return CLIP_LABEL_MAP[best_idx]


def compute_attention_rollout(outputs, discard_ratio=0.9):
    attentions  = outputs.attentions
    num_patches = int(math.sqrt(attentions[0].shape[-1] - 1))
    result      = torch.eye(attentions[0].shape[-1])
    for attn in attentions:
        a      = attn.squeeze(0).mean(dim=0)
        a      = a + torch.eye(a.shape[0])
        a      = a / a.sum(dim=-1, keepdim=True)
        flat   = a.flatten()
        thresh = flat.kthvalue(int(discard_ratio * flat.numel())).values
        a[a < thresh] = 0.0
        result = torch.matmul(a, result)
    mask = result[0, 1:].detach().cpu()
    mask = mask.reshape(num_patches, num_patches).numpy()
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return mask


def get_region_bbox(region_name, image_type, img_w, img_h, rollout_shape):
    labels = GRID_LABELS.get(image_type, GRID_LABELS["face"])
    for ri in range(3):
        for ci in range(3):
            if labels[ri][ci] == region_name:
                return (ci * (img_w // 3), ri * (img_h // 3), img_w // 3, img_h // 3)
    return (0, 0, img_w, img_h)


def make_centered_region_crop(pil_image, bx, by, bw, bh, crop_size=200):
    img_w, img_h = pil_image.size
    half = crop_size // 2
    cx, cy = bx + bw // 2, by + bh // 2
    x0, y0 = cx - half, cy - half
    x1, y1 = x0 + crop_size, y0 + crop_size
    canvas  = Image.new("RGB", (crop_size, crop_size), (255, 255, 255))
    src_x0, src_y0 = max(x0, 0), max(y0, 0)
    src_x1, src_y1 = min(x1, img_w), min(y1, img_h)
    if src_x1 > src_x0 and src_y1 > src_y0:
        patch = pil_image.crop((src_x0, src_y0, src_x1, src_y1))
        canvas.paste(patch, (src_x0 - x0, src_y0 - y0))
    draw = ImageDraw.Draw(canvas)
    rx0, ry0 = bx - x0, by - y0
    rx1, ry1 = rx0 + bw, ry0 + bh
    rx0, ry0 = max(rx0, 2), max(ry0, 2)
    rx1, ry1 = min(rx1, crop_size - 2), min(ry1, crop_size - 2)
    if rx1 > rx0 and ry1 > ry0:
        for t in range(3):
            draw.rectangle([rx0 + t, ry0 + t, rx1 - t, ry1 - t], outline=(239, 68, 68), fill=None)
    return canvas


def pil_to_base64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode()


def heatmap_to_base64(heatmap_2d, pil_image_resized, display_size, cmap_name="Reds"):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(pil_image_resized, alpha=0.3, cmap="gray")
    gs  = heatmap_2d.shape[0]
    x   = np.linspace(0, display_size[0], gs + 1)
    y   = np.linspace(0, display_size[1], gs + 1)
    ax.pcolormesh(*np.meshgrid(x, y), heatmap_2d, cmap=cmap_name, shading="auto", alpha=0.7)
    ax.axis("off")
    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=100)
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def label_regions_grid(rollout, image_type):
    h, w   = rollout.shape
    labels = GRID_LABELS.get(image_type, GRID_LABELS["face"])
    scores = {}
    rh, cw = h // 3, w // 3
    for ri in range(3):
        for ci in range(3):
            lbl   = labels[ri][ci]
            patch = rollout[ri * rh:(ri + 1) * rh, ci * cw:(ci + 1) * cw]
            scores.setdefault(lbl, []).append(float(patch.mean()))
    scores = {k: float(np.mean(v)) for k, v in scores.items()}
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]


def generate_clues_rule_based(top_regions, label, confidence, pil_image, image_type, rollout_shape):
    icons  = ["👁", "✶", "◈"]
    img_w, img_h = pil_image.size
    clues  = []
    for i, (region_name, score) in enumerate(top_regions):
        desc       = REGION_DESCRIPTIONS.get(region_name, (region_name, "This area showed unusual patterns."))
        short_name, explanation = desc
        intensity  = "strongly" if score > 0.6 else ("noticeably" if score > 0.35 else "slightly")
        bx, by, bw, bh = get_region_bbox(region_name, image_type, img_w, img_h, rollout_shape)
        crop = make_centered_region_crop(pil_image, bx, by, bw, bh, crop_size=200)
        clues.append({
            "icon":   icons[i % len(icons)],
            "region": region_name,
            "clue":   f"The model focused {intensity} on {short_name}. {explanation}",
            "crop":   pil_to_base64(crop),
        })
    return clues


def generate_verdict_sentence(label, confidence):
    certainty = "very likely" if confidence >= 0.85 else ("likely" if confidence >= 0.65 else "possibly")
    action    = "created by an AI tool" if label == "DEEPFAKE" else "a real photograph"
    return f"Our model believes this image is {certainty} {action} ({int(confidence * 100)}% confidence)."


# ------------------------------------------------------------------
# 6. API ENDPOINTS — Text Pipeline
# ------------------------------------------------------------------
@app.post("/predict")
def predict(request: TextRequest):
    """Pipeline Stage 1: Fast classification inference only (<150ms)."""
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        if device == "cuda":
            with torch.amp.autocast("cuda"):
                outputs = model_txt(**inputs)
        else:
            outputs = model_txt(**inputs)
    probs      = torch.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probs, dim=-1).item()
    label      = "AI-Generated" if prediction == 1 else "Human-Written"
    confidence = round(float(probs[0][prediction]), 4)
    return {"prediction": label, "confidence": confidence}


@app.post("/explain-tokens")
def explain_tokens(request: TextRequest):
    """Pipeline Stage 2: Background XAI Integrated Gradients."""
    token_attributions = run_integrated_gradients(request.text, pred_class=1, steps=2, internal_batch_size=2)
    return {"tokens": token_attributions}


@app.post("/explain-linguistics")
def explain_linguistics(payload: TokenExplanationRequest):
    """Pipeline Stage 3: LLM-powered user-friendly linguistic insight."""
    try:
        client   = get_openai_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": payload.json()}
            ],
            response_format={"type": "json_object"},
            temperature=0.5,
        )
        llm_output  = json.loads(response.choices[0].message.content)
        explanation = llm_output.get("explanation", "Linguistic patterns verified.")
    except Exception as e:
        explanation = f"Linguistic insight processing engine unavailable. ({str(e)})"
    return {"explanation": explanation}


# ------------------------------------------------------------------
# 7. API ENDPOINTS — Image Pipeline
# ------------------------------------------------------------------
@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    """Fast deepfake prediction (no explainability)."""
    image  = Image.open(io.BytesIO(await file.read())).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model_img(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    return {"label": "Fake" if prediction == 0 else "Real"}


@app.post("/explain-image")
async def explain_image(file: UploadFile = File(...)):
    """Full explainability report: CLIP content type + attention rollout + semantic clues."""
    image        = Image.open(io.BytesIO(await file.read())).convert("RGB")
    display_size = (400, 400)
    image_resized = image.resize(display_size, Image.Resampling.LANCZOS)

    # 1. CLIP content classification
    image_type = classify_image_type_clip(image)

    # 2. Multi-crop ensemble prediction with attention rollout on centre crop
    w, h  = image.size
    short = min(w, h)
    crops = [
        image.crop(((w - short) // 2, (h - short) // 2, (w + short) // 2, (h + short) // 2)),
        image.crop((0,       0,       short, short)),
        image.crop((w - short, 0,     w,     short)),
        image.crop((0,       h - short, short, h)),
        image.crop((w - short, h - short, w, h)),
    ]
    all_probs, main_outputs = [], None
    for i, crop in enumerate(crops):
        inputs = explainer_processor(images=crop, return_tensors="pt").to(device)
        with torch.no_grad():
            out = explainer_model(**inputs, output_attentions=(i == 0))
            if i == 0:
                main_outputs = out
            all_probs.append(F.softmax(out.logits, dim=-1)[0].cpu().numpy())

    avg_probs  = np.mean(all_probs, axis=0)
    fake_prob, real_prob = float(avg_probs[0]), float(avg_probs[1])
    prediction = 0 if fake_prob > real_prob else 1
    label      = "DEEPFAKE" if prediction == 0 else "REAL"
    confidence = float(avg_probs[prediction])

    # 3. Attention rollout heatmap
    rollout    = compute_attention_rollout(main_outputs)
    cmap_name  = "Reds" if prediction == 0 else "Greens"
    heatmap_b64 = heatmap_to_base64(rollout, image_resized, display_size, cmap_name)

    # 4. Semantic region labelling and clues
    top_regions     = label_regions_grid(rollout, image_type)
    clues           = generate_clues_rule_based(top_regions, label, confidence, image, image_type, rollout.shape)
    analogy         = ANALOGY_BANK.get(image_type, ANALOGY_BANK["face"])
    verdict_sentence = generate_verdict_sentence(label, confidence)

    return {
        "verdict":               label,
        "confidence":            confidence,
        "fake_prob":             fake_prob,
        "real_prob":             real_prob,
        "image_type":            image_type,
        "original_image_base64": pil_to_base64(image_resized),
        "heatmap_base64":        heatmap_b64,
        "verdict_sentence":      verdict_sentence,
        "clues":                 clues,
        "analogy":               analogy,
    }


# ------------------------------------------------------------------
# 8. HEALTH CHECK & FACT-CHECKING ENDPOINTS
# ------------------------------------------------------------------
@app.get("/", tags=["Health"])
def root():
    return {
        "status":  "ok",
        "service": "Misinformation & Deepfake Detection API",
        "model":   agent.model_name,
        "date":    agent.get_current_date_info()["full_date"],
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