"""
FastAPI Backend for beeHIVe Molecular Applications
Provides endpoints for molecule generation, visualization, and HIV prediction
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

app = FastAPI(
    title="beeHIVe Molecular API",
    description="FastAPI backend for molecule generation, visualization, and prediction",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class GenerateMoleculesRequest(BaseModel):
    num_samples: int = 100
    temperature: float = 1.0
    edge_threshold: float = 0.5

class GenerateMoleculesResponse(BaseModel):
    success: bool
    molecules_count: int
    validity_rate: float
    uniqueness_rate: float
    message: str
    smiles_file: Optional[str] = None

class VisualizeMoleculeRequest(BaseModel):
    smiles: str
    width: int = 400
    height: int = 400
    format: str = "PNG"

class Generate3DMoleculeRequest(BaseModel):
    smiles: str
    optimize: bool = True
    force_field: str = "MMFF"  # MMFF or UFF

class VisualizeMoleculeResponse(BaseModel):
    success: bool
    smiles: str
    image: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class Generate3DMoleculeResponse(BaseModel):
    success: bool
    smiles: str
    pdb: Optional[str] = None
    sdf: Optional[str] = None
    xyz: Optional[str] = None
    num_atoms: int = 0
    optimized: bool = False
    force_field: Optional[str] = None
    error: Optional[str] = None

class PredictHIVRequest(BaseModel):
    smiles: str

class PredictHIVResponse(BaseModel):
    success: bool
    smiles: str
    prediction: Optional[int] = None  # Binary prediction (0 or 1)
    probability: Optional[float] = None  # Raw probability score
    effectiveness_score: Optional[float] = None  # Effectiveness score (0-100)
    category: Optional[str] = None  # Effectiveness category
    descriptors: Optional[Dict[str, Any]] = None  # Additional molecular descriptors
    error: Optional[str] = None

class MoleculeInfo(BaseModel):
    smiles: str
    index: int
    properties: Optional[Dict[str, Any]] = None

class GetMoleculesResponse(BaseModel):
    success: bool
    molecules: List[MoleculeInfo]
    count: int
    message: Optional[str] = None

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "beeHIVe Molecular API",
        "version": "1.0.0",
                "endpoints": {
            "generate_molecules": "/api/generate-molecules",
            "get_molecules": "/api/get-molecules",
            "visualize_molecule": "/api/visualize-molecule",
            "generate_3d_molecule": "/api/generate-3d-molecule",
            "predict_hiv": "/api/predict-hiv"
        }
    }

@app.post("/api/generate-molecules", response_model=GenerateMoleculesResponse)
async def generate_molecules(request: GenerateMoleculesRequest):
    """Generate new molecules using GraphVAE"""
    try:
        # Path to the GraphVAE script
        script_path = project_root / "GraphVAE" / "vae_generate_molecule.py"
        
        if not script_path.exists():
            raise HTTPException(
                status_code=404, 
                detail="GraphVAE generation script not found"
            )
        
        # Create a temporary script to run generation with parameters
        temp_script_content = f"""
import sys
sys.path.append('{project_root / "GraphVAE"}')

from vae_generate_molecule import Generator
import torch
import os

# Change to GraphVAE directory
os.chdir('{project_root / "GraphVAE"}')

# Find model
model_path = None
for path in ['vae_best_model.pth', 'vae_final_model.pth']:
    if os.path.exists(path):
        model_path = path
        break

if not model_path:
    print("ERROR: No model found")
    exit(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = Generator(model_path, device)

molecules, smiles_list, validity_rate, uniqueness_rate = generator.generate_molecules(
    num_samples={request.num_samples},
    temperature={request.temperature},
    edge_threshold={request.edge_threshold}
)

# Filter valid molecules
valid_smiles = [smiles for smiles in smiles_list if smiles is not None]

# Save to file
with open('vae_generated_smiles.txt', 'w') as f:
    for smiles in valid_smiles:
        f.write(f"{{smiles}}\\n")

print(f"RESULT: {{len(valid_smiles)}} {{validity_rate:.4f}} {{uniqueness_rate:.4f}}")
"""
        
        # Write and execute temporary script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(temp_script_content)
            temp_script_path = f.name
        
        # Initialize variables with default values
        molecules_count = 0
        validity_rate = 0.0
        uniqueness_rate = 0.0
        
        try:
            # Run the generation script
            result = subprocess.run(
                [sys.executable, temp_script_path],
                capture_output=True,
                text=True,
                cwd=project_root / "GraphVAE",
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                raise HTTPException(
                    status_code=500,
                    detail=f"Generation failed: {result.stderr}"
                )
            
            # Parse result
            output_lines = result.stdout.strip().split('\n')
            result_line = None
            for line in output_lines:
                if line.startswith("RESULT:"):
                    result_line = line
                    break
            
            if result_line:
                try:
                    parts = result_line.replace("RESULT: ", "").split()
                    if len(parts) >= 3:
                        molecules_count = int(parts[0])
                        validity_rate = float(parts[1])
                        uniqueness_rate = float(parts[2])
                except (ValueError, IndexError) as e:
                    print(f"Error parsing result line: {result_line}, error: {e}")
                    # Keep default values
            
            # Check if output file was created
            smiles_file_path = project_root / "GraphVAE" / "vae_generated_smiles.txt"
            smiles_file = str(smiles_file_path) if smiles_file_path.exists() else None
            
            return GenerateMoleculesResponse(
                success=True,
                molecules_count=molecules_count,
                validity_rate=validity_rate,
                uniqueness_rate=uniqueness_rate,
                message=f"Successfully generated {molecules_count} molecules",
                smiles_file=smiles_file
            )
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_script_path):
                os.unlink(temp_script_path)
            
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=408,
            detail="Molecule generation timed out"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating molecules: {str(e)}"
        )

@app.get("/api/get-molecules", response_model=GetMoleculesResponse)
async def get_molecules():
    """Get previously generated molecules"""
    try:
        smiles_file_path = project_root / "GraphVAE" / "vae_generated_smiles.txt"
        
        if not smiles_file_path.exists():
            return GetMoleculesResponse(
                success=True,
                molecules=[],
                count=0,
                message="No molecules generated yet"
            )
        
        # Read SMILES file
        with open(smiles_file_path, 'r') as f:
            smiles_list = [line.strip() for line in f if line.strip()]
        
        # Create molecule objects
        molecules = []
        for i, smiles in enumerate(smiles_list):
            molecules.append(MoleculeInfo(
                smiles=smiles,
                index=i,
                properties=None  # Properties will be calculated by visualization endpoint
            ))
        
        return GetMoleculesResponse(
            success=True,
            molecules=molecules,
            count=len(molecules),
            message=f"Retrieved {len(molecules)} molecules"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading molecules: {str(e)}"
        )

@app.post("/api/visualize-molecule", response_model=VisualizeMoleculeResponse)
async def visualize_molecule(request: VisualizeMoleculeRequest):
    """Visualize a molecule from SMILES string"""
    try:
        # Import the molecule visualizer
        sys.path.append(str(project_root / "GraphVAE"))
        from molecule_visualizer import visualize_single_molecule
        
        # Generate visualization
        result = visualize_single_molecule(
            smiles=request.smiles,
            output_path=None,
            format=request.format,
            width=request.width,
            height=request.height
        )
        
        if result['success']:
            return VisualizeMoleculeResponse(
                success=True,
                smiles=request.smiles,
                image=result['image'],
                properties=result['properties']
            )
        else:
            return VisualizeMoleculeResponse(
                success=False,
                smiles=request.smiles,
                error=result['error']
            )
            
    except Exception as e:
        return VisualizeMoleculeResponse(
            success=False,
            smiles=request.smiles,
            error=f"Visualization error: {str(e)}"
        )

@app.post("/api/generate-3d-molecule", response_model=Generate3DMoleculeResponse)
async def generate_3d_molecule(request: Generate3DMoleculeRequest):
    """Generate 3D molecule structure with coordinates in PDB/SDF/XYZ formats"""
    try:
        # Import the molecule visualizer with 3D functionality
        sys.path.append(str(project_root / "GraphVAE"))
        from molecule_visualizer import generate_3d_molecule_data
        
        # Generate 3D structure
        result = generate_3d_molecule_data(
            smiles=request.smiles,
            optimize=request.optimize,
            force_field=request.force_field
        )
        
        if result['success']:
            return Generate3DMoleculeResponse(
                success=True,
                smiles=request.smiles,
                pdb=result['pdb'],
                sdf=result['sdf'],
                xyz=result['xyz'],
                num_atoms=result['num_atoms'],
                optimized=result['optimized'],
                force_field=result['force_field']
            )
        else:
            return Generate3DMoleculeResponse(
                success=False,
                smiles=request.smiles,
                error=result['error']
            )
            
    except Exception as e:
        return Generate3DMoleculeResponse(
            success=False,
            smiles=request.smiles,
            error=f"3D generation error: {str(e)}"
        )

@app.post("/api/predict-hiv", response_model=PredictHIVResponse)
async def predict_hiv(request: PredictHIVRequest):
    """Predict HIV activity for a molecule using the predict.py script"""
    try:
        # Path to the predict.py script
        predict_script_path = project_root / "predict.py"
        
        if not predict_script_path.exists():
            raise HTTPException(
                status_code=404, 
                detail="Prediction script not found"
            )
        
        # Create a temporary script to call predict.py with the SMILES
        temp_script_content = f"""
import sys
import os
sys.path.append('{project_root}')

# Change to project root directory
os.chdir('{project_root}')

try:
    from predict import main
    result = main('{request.smiles}')
    
    if result:
        print(f"SUCCESS: {{result}}")
    else:
        print("ERROR: No result returned")
        exit(1)
        
except Exception as e:
    print(f"ERROR: {{str(e)}}")
    exit(1)
"""
        
        # Write and execute temporary script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(temp_script_content)
            temp_script_path = f.name
        
        try:
            # Run the prediction script
            result = subprocess.run(
                [sys.executable, temp_script_path],
                capture_output=True,
                text=True,
                cwd=project_root,
                timeout=120  # 2 minute timeout for model loading
            )
            
            if result.returncode != 0:
                error_msg = result.stderr or "Unknown prediction error"
                return PredictHIVResponse(
                    success=False,
                    smiles=request.smiles,
                    error=f"Prediction failed: {error_msg}"
                )
            
            # Parse result from output
            output_lines = result.stdout.strip().split('\n')
            result_data = None
            
            for line in output_lines:
                if line.startswith("SUCCESS: "):
                    try:
                        # Extract the dictionary string and parse it
                        result_str = line.replace("SUCCESS: ", "")
                        result_data = eval(result_str)  # Safe because we control the output format
                        break
                    except Exception as e:
                        print(f"Error parsing result: {e}")
                        continue
            
            if result_data and isinstance(result_data, dict):
                # Convert boolean to int for prediction
                prediction = 1 if result_data.get('is_hiv_active', False) else 0
                
                return PredictHIVResponse(
                    success=True,
                    smiles=request.smiles,
                    prediction=prediction,
                    probability=result_data.get('probability', 0.0),
                    effectiveness_score=result_data.get('effectiveness_score', 0.0),
                    category=result_data.get('category', 'Unknown')
                )
            else:
                return PredictHIVResponse(
                    success=False,
                    smiles=request.smiles,
                    error="Could not parse prediction result"
                )
                
        finally:
            # Clean up temporary file
            if os.path.exists(temp_script_path):
                os.unlink(temp_script_path)
            
    except subprocess.TimeoutExpired:
        return PredictHIVResponse(
            success=False,
            smiles=request.smiles,
            error="Prediction timed out"
        )
    except Exception as e:
        return PredictHIVResponse(
            success=False,
            smiles=request.smiles,
            error=f"Error predicting HIV activity: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "FastAPI backend is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
