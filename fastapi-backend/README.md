# OverFiT FastAPI Backend

A FastAPI backend for molecular generation, visualization, and HIV activity prediction.

## Features

- **Molecule Generation**: Generate new molecules using GraphVAE models
- **Molecule Visualization**: Create visual representations of molecules from SMILES strings
- **HIV Activity Prediction**: Predict HIV activity using ensemble models
- **RESTful API**: Well-documented API endpoints with automatic OpenAPI/Swagger documentation

## Installation

1. **Clone the repository** (if not already done):

   ```bash
   cd /Users/hydrogen/Desktop/OverFiT
   ```

2. **Create a virtual environment**:

   ```bash
   python -m venv fastapi-env
   source fastapi-env/bin/activate  # On Windows: fastapi-env\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   cd fastapi-backend
   pip install -r requirements.txt
   ```

4. **Ensure model files are available**:
   - GraphVAE models should be in `../GraphVAE/` directory
   - HIV prediction models should be in their respective directories
   - Run training scripts if models are not available

## Usage

### Starting the Server

```bash
cd fastapi-backend
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### API Documentation

- **Interactive API docs**: http://localhost:8000/docs
- **Alternative docs**: http://localhost:8000/redoc
- **OpenAPI schema**: http://localhost:8000/openapi.json

## API Endpoints

### 1. Generate Molecules

- **Endpoint**: `POST /api/generate-molecules`
- **Description**: Generate new molecules using GraphVAE
- **Request Body**:
  ```json
  {
    "num_samples": 100,
    "temperature": 1.0,
    "edge_threshold": 0.5
  }
  ```

### 2. Get Generated Molecules

- **Endpoint**: `GET /api/get-molecules`
- **Description**: Retrieve previously generated molecules
- **Response**: List of molecules with SMILES strings

### 3. Visualize Molecule

- **Endpoint**: `POST /api/visualize-molecule`
- **Description**: Create visual representation of a molecule
- **Request Body**:
  ```json
  {
    "smiles": "CCO",
    "width": 400,
    "height": 400,
    "format": "PNG"
  }
  ```

### 4. Predict HIV Activity

- **Endpoint**: `POST /api/predict-hiv`
- **Description**: Predict HIV activity for a molecule
- **Request Body**:
  ```json
  {
    "smiles": "CCO"
  }
  ```

### 5. Health Check

- **Endpoint**: `GET /health`
- **Description**: Check if the API is running

## Integration with Next.js Frontend

The FastAPI backend is designed to work with the existing Next.js frontend. The API endpoints mirror the structure expected by the frontend routes in `overfit-app/src/app/api/`.

### CORS Configuration

The backend includes CORS middleware configured to allow requests from the Next.js frontend. For production deployment, update the `allow_origins` setting in `main.py`.

## Development

### Project Structure

```
fastapi-backend/
├── main.py              # Main FastAPI application
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── .env               # Environment variables (create as needed)
```

### Environment Variables

Create a `.env` file for configuration:

```bash
# Model paths (optional, defaults to relative paths)
GRAPHVAE_MODEL_PATH=../GraphVAE/
HIV_MODEL_PATH=../Complete/

# API configuration
HOST=0.0.0.0
PORT=8000
DEBUG=True
```

### Adding New Endpoints

1. Define Pydantic models for request/response
2. Create endpoint function with appropriate decorators
3. Add error handling and validation
4. Update documentation

## Troubleshooting

### Common Issues

1. **Module Import Errors**: Ensure all required packages are installed and the virtual environment is activated
2. **Model Not Found**: Check that model files exist in expected locations
3. **RDKit Installation**: RDKit can be tricky to install; consider using conda if pip fails
4. **CUDA Issues**: PyTorch may need CUDA-specific installation for GPU support

### Error Logs

Check the console output for detailed error messages. All endpoints include comprehensive error handling with descriptive messages.

## Production Deployment

For production deployment, consider:

1. **Use a production ASGI server**:

   ```bash
   pip install gunicorn
   gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
   ```

2. **Set up a reverse proxy** (nginx, Apache, etc.)

3. **Configure environment variables** for production settings

4. **Enable HTTPS** for secure communication

## Dependencies

- **FastAPI**: Modern, fast web framework for building APIs
- **PyTorch**: Deep learning framework for neural network models
- **RDKit**: Chemistry toolkit for molecular operations
- **Uvicorn**: ASGI server for running FastAPI applications

## License

This project follows the same license as the main OverFiT repository.
