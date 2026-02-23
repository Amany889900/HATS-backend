import os
import zipfile
import subprocess

print("Downloading model...")

subprocess.run([
    "kaggle",
    "datasets",
    "download",
    "-d",
    "takwasobhy/final-finetuned-vit"
])

print("Extracting model...")

with zipfile.ZipFile(
    "final-finetuned-vit.zip", 'r'
) as zip_ref:
    zip_ref.extractall("./models/vit-deepfake")

print("Done!")