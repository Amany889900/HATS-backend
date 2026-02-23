# HATS Backend: Human & AI Truth Scanner

This is the Python-based AI backend for the HATS project. It uses **FastAPI** to serve two models: a text-based AI detector and a Vision Transformer (ViT) for deepfake image detection.

## 🚀 Getting Started

### 1. Clone and Setup Environment

```bash
# Clone the repo
git clone <your-repo-url>
cd HATS-backend

# Create a virtual environment
python -m venv .venv

# Activate the environment
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate

```

### 2. Install Dependencies

Once the environment is active, install the required libraries:

```bash
pip install -r requirements.txt

```

### 3. Setup the Models (Crucial)

Because the models are too large for GitHub (>4GB total), you must set them up locally:

* **Text Model:** This will auto-download the first time you run the server (approx. 1.7GB).
* **Image Model (2.4GB):** 1. Run the download script: `python download_model.py`.
2. Ensure the files are extracted to: `./models/vit-deepfake/`.

### 4. Run the Server

```bash
uvicorn main:app --reload

```

The server will be live at `http://127.0.0.1:8000`. You can test the endpoints directly at `http://127.0.0.1:8000/docs`.

## 🛠 API Endpoints

* `POST /predict`: Sends text to the AI Text Detector.
* `POST /predict-image`: Sends an image file to the Deepfake Detector.

