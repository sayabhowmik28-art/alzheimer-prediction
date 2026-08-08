import io
import os
import sys

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.dataset import get_transforms
from src.model import build_model

app = FastAPI(title="Alzheimer's MRI Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CHECKPOINT_PATH = os.environ.get(
    "CHECKPOINT_PATH",
    os.path.join(config.OUTPUT_DIR, config.CHECKPOINT_NAME),
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None
CLASS_NAMES = None
TRANSFORM = get_transforms(train=False)


# --- Content shown alongside each predicted stage ---
STAGE_INFO = {
    "NonDemented": {
        "stage_display": "Non-Demented",
        "urgency": "Low",
        "summary": "The scan's structural pattern most closely resembles the non-demented reference group, with no strong indicators of atrophy typically associated with Alzheimer's progression.",
        "typical_symptoms": [
            "No noticeable memory or cognitive changes",
            "Normal day-to-day functioning",
        ],
        "recommended_actions": [
            "No immediate action required based on this result alone",
            "Continue routine health checkups",
        ],
        "follow_up_imaging": "Not typically needed unless new symptoms appear.",
    },
    "VeryMildDemented": {
        "stage_display": "Very Mild Demented",
        "urgency": "Moderate",
        "summary": "The scan shows subtle structural patterns consistent with very early, mild cognitive changes. This stage can sometimes reflect normal aging rather than disease.",
        "typical_symptoms": [
            "Occasional forgetfulness (names, recent conversations)",
            "Minor difficulty finding words",
        ],
        "recommended_actions": [
            "Discuss this result with a physician or neurologist",
            "Consider a baseline cognitive assessment",
        ],
        "follow_up_imaging": "Repeat imaging in 6-12 months is often recommended to track any change.",
    },
    "MildDemented": {
        "stage_display": "Mild Demented",
        "urgency": "High",
        "summary": "The scan shows structural patterns consistent with mild dementia, including changes commonly associated with early Alzheimer's disease.",
        "typical_symptoms": [
            "Noticeable memory lapses affecting daily life",
            "Difficulty with planning or problem-solving",
            "Getting lost in familiar places",
        ],
        "recommended_actions": [
            "Schedule an evaluation with a neurologist promptly",
            "Consider a full cognitive/neuropsychological assessment",
        ],
        "follow_up_imaging": "Follow-up imaging within 3-6 months is commonly advised.",
    },
    "ModerateDemented": {
        "stage_display": "Moderate Demented",
        "urgency": "Very High",
        "summary": "The scan shows significant structural changes consistent with moderate dementia, where daily functioning is typically substantially affected.",
        "typical_symptoms": [
            "Significant memory loss, including of personal history",
            "Needing help with daily activities",
            "Confusion about time or place",
        ],
        "recommended_actions": [
            "Seek prompt evaluation and care planning with a neurologist",
            "Discuss support and caregiving resources with a physician",
        ],
        "follow_up_imaging": "Closer, physician-directed monitoring is typically recommended.",
    },
}


@app.on_event("startup")
def load_model():
    """Load the checkpoint once when the server starts, not on every request."""
    global MODEL, CLASS_NAMES

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"WARNING: no checkpoint found at {CHECKPOINT_PATH}. "
              f"Train a model first (see README) or set CHECKPOINT_PATH. "
              f"/api/predict-mri will return 503 until a checkpoint is available.")
        return

    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    CLASS_NAMES = ckpt["class_names"]
    model = build_model(ckpt["backbone"], num_classes=len(CLASS_NAMES), img_size=config.IMG_SIZE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()
    MODEL = model
    print(f"Loaded checkpoint '{CHECKPOINT_PATH}' | classes: {CLASS_NAMES} | device: {DEVICE}")


# Make HTTPException error bodies look like {"error": "..."} instead of
# FastAPI's default {"detail": "..."} — sayan.html reads result.data.error.
@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.get("/api/status")
def status():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "mri_model_available": MODEL is not None,
        "device": str(DEVICE),
    }


@app.get("/api/health")
def health():
    return status()


@app.post("/api/predict-mri")
async def predict_mri(image: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="No trained model checkpoint is loaded on the server yet.",
        )

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file (jpg/png).")

    raw = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded file as an image.")

    tensor = TRANSFORM(pil_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = MODEL(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()

    ranked = sorted(zip(CLASS_NAMES, probs), key=lambda x: x[1], reverse=True)
    probabilities = {label: prob for label, prob in zip(CLASS_NAMES, probs)}

    predicted_class = ranked[0][0]
    info = STAGE_INFO.get(predicted_class, {})

    return {
        "prediction": predicted_class,
        "stage_key": predicted_class,
        "stage_display": info.get("stage_display", predicted_class),
        "confidence": ranked[0][1],
        "probabilities": probabilities,
        "all_classes": [{"label": label, "probability": prob} for label, prob in ranked],
        "urgency": info.get("urgency", "Unknown"),
        "summary": info.get("summary", ""),
        "typical_symptoms": info.get("typical_symptoms", []),
        "recommended_actions": info.get("recommended_actions", []),
        "follow_up_imaging": info.get("follow_up_imaging", ""),
    }


# --- Also keep /api/predict as an alias, in case anything else calls it ---
@app.post("/api/predict")
async def predict(image: UploadFile = File(...)):
    return await predict_mri(image)


# --- Serve the frontend ---
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")

if os.path.isdir(FRONTEND_DIR):
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(FRONTEND_DIR, "sayan.html"))