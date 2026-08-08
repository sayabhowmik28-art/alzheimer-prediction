CLASS_NAMES = [
    "NonDemented",
    "VeryMildDemented",
    "MildDemented",
    "ModerateDemented",
]
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 224          # resize target (224 matches ImageNet backbones)
VAL_SPLIT = 0.15        # fraction of the training folder held out for validation

# ---- Training ----
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 6
NUM_WORKERS = 4
SEED = 42

# ---- Normalization (ImageNet stats — used whenever backbone is pretrained) ----
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# ---- Paths ----
OUTPUT_DIR = "outputs"
CHECKPOINT_NAME = "best_model.pt"
