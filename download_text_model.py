from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# Define the model name and local path
MODEL_NAME = "abhi099k/ai-text-detector-L0"
SAVE_PATH = "./models/text-detector"

# Create the directory if it doesn't exist
os.makedirs(SAVE_PATH, exist_ok=True)

print(f"Downloading {MODEL_NAME}...")

# Download and save the tokenizer and model locally
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained(SAVE_PATH)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.save_pretrained(SAVE_PATH)

print(f"Done! Model saved to {SAVE_PATH}")