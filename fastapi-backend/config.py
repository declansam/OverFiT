"""
Configuration settings for the FastAPI backend
"""

import os
from pathlib import Path
from typing import List

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Server settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
DEBUG = os.getenv("DEBUG", "True").lower() == "true"

# Model paths
GRAPHVAE_MODEL_PATH = Path(os.getenv("GRAPHVAE_MODEL_PATH", PROJECT_ROOT / "GraphVAE"))
HIV_MODEL_PATH = Path(os.getenv("HIV_MODEL_PATH", PROJECT_ROOT / "Complete"))
MORGAN_MLP_PATH = Path(os.getenv("MORGAN_MLP_PATH", PROJECT_ROOT / "MorganFingerprintMLP"))
STRUCTURAL_MODEL_PATH = Path(os.getenv("STRUCTURAL_MODEL_PATH", PROJECT_ROOT / "StructuralModels"))
GNN_MODEL_PATH = Path(os.getenv("GNN_MODEL_PATH", PROJECT_ROOT / "GNNModels"))

# API settings
MAX_MOLECULES = int(os.getenv("MAX_MOLECULES", 1000))
GENERATION_TIMEOUT = int(os.getenv("GENERATION_TIMEOUT", 300))  # 5 minutes
PREDICTION_TIMEOUT = int(os.getenv("PREDICTION_TIMEOUT", 60))   # 1 minute

# CORS settings
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000"
]

# Add environment-specific origins
if os.getenv("ALLOWED_ORIGINS"):
    ALLOWED_ORIGINS.extend(os.getenv("ALLOWED_ORIGINS").split(","))

# File paths
GENERATED_SMILES_FILE = GRAPHVAE_MODEL_PATH / "vae_generated_smiles.txt"
MOLECULE_PROPERTIES_FILE = GRAPHVAE_MODEL_PATH / "vae_molecule_properties.csv"
