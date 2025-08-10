# beeHIVe - Molecular Discovery Platform

A full-stack application for molecular generation, visualization, and HIV activity prediction using machine learning.

## Components

### 🚀 Next.js Frontend (`overfit-app/`)

Modern web interface for molecular discovery and analysis.

**Quick Start:**

```bash
cd overfit-app
npm install
npm run dev
```

Visit `http://localhost:3000`

**Features:**

- Generate molecules using GraphVAE
- Visualize molecular structures
- Predict HIV activity
- Interactive 3D molecule viewer

### ⚡ FastAPI Backend (`fastapi-backend/`)

High-performance API server with machine learning endpoints.

**Quick Start:**

```bash
cd fastapi-backend
pip install -r requirements.txt
python run.py
```

API available at `http://localhost:8000`

**Endpoints:**

- `/api/generate-molecules` - Generate new molecules
- `/api/visualize-molecule` - Create molecular visualizations
- `/api/predict-hiv` - Predict HIV activity
- `/api/generate-3d-molecule` - Generate 3D structures

## Tech Stack

- **Frontend:** Next.js 15, React 19, TypeScript, Tailwind CSS
- **Backend:** FastAPI, PyTorch, RDKit, scikit-learn
- **ML Models:** GraphVAE, GNN, MLP classifiers
