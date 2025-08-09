"""
Prediction functionality for trained GNN models
"""

import torch
from ogb.graphproppred import PygGraphPropPredDataset
from models import HybridGNN, GINConvNet
from datasets import setup_pytorch_compatibility


def predict_molecule(model, molecule_data, device, config):
    """
    Predict HIV activity for a single molecule

    Args:
        model: Trained GNN model
        molecule_data: Input data for the molecule
        device: Device to run the model on
        config: Configuration object containing prediction parameters

    Returns: 
        prediction (0/1), confidence score, effectiveness ranking
    """
    
    threshold = config.get('prediction.threshold', 0.5)

    model.eval()

    with torch.no_grad():
        molecule_data = molecule_data.to(device)

        # Ensure features are float
        if hasattr(molecule_data, 'x'):
            molecule_data.x = molecule_data.x.float()

        # Get raw prediction score
        out = model(molecule_data.x, molecule_data.edge_index,
                   torch.zeros(molecule_data.x.size(0), dtype=torch.long, device=device))

        # Apply sigmoid to get probability
        prob = torch.sigmoid(out).item()

        # Binary classification
        prediction = 1 if prob > threshold else 0

        # Effectiveness ranking (0-100 scale)
        effectiveness_score = prob * 100

        # Determine effectiveness category
        if effectiveness_score >= 80:
            category = "Highly Effective"
        elif effectiveness_score >= 60:
            category = "Moderately Effective"
        elif effectiveness_score >= 40:
            category = "Weakly Effective"
        elif effectiveness_score >= 20:
            category = "Minimally Effective"
        else:
            category = "Not Effective"

    return {
        'prediction': prediction,
        'probability': prob,
        'effectiveness_score': effectiveness_score,
        'category': category,
        'is_hiv_active': bool(prediction)
    }


def load_trained_model(config, num_features, num_classes, device):
    """
    Load a trained model from file

    Args:
        config: Configuration object containing model parameters
        num_features: Number of input features
        num_classes: Number of output classes
        device: Device to load the model on

    Returns:
        Trained GNN model or None
    """
    
    model_path = config.get('prediction.model_save_path', 'best_hiv_gnn_model.pth')
    model_type = config.get('model.type', 'hybrid')
    hidden_dim = config.get('model.hidden_dim', 256)
    dropout = config.get('model.dropout', 0.3)
    num_layers = config.get('model.num_layers', 5)
    
    # Initialize model
    if model_type == 'hybrid':
        model = HybridGNN(
            num_features=num_features,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
    elif model_type == 'gin':
        model = GINConvNet(
            num_features=num_features,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)

    # Load trained weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print("Model loaded successfully!")
    except:
        try:
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    return model


def demo_prediction(config, device):
    """
    Demo function to show how to use the trained model for prediction
    """
    
    dataset_name = config.get('dataset.name', 'ogbg-molhiv')
    dataset_root = config.get('dataset.root', 'datasets/')
    test_indices = config.get('demo.test_indices', [0, 100, 200, 300, 400])

    # Load the dataset to get a sample molecule
    print("Loading dataset for demo...")
    setup_pytorch_compatibility()
    
    try:
        dataset = PygGraphPropPredDataset(name=dataset_name, root=dataset_root)
    except Exception as e:
        if "weights_only" in str(e):
            original_load = torch.load
            torch.load = lambda *args, **kwargs: original_load(*args, **{k: v for k, v in kwargs.items() if k != 'weights_only'}, weights_only=False)
            dataset = PygGraphPropPredDataset(name=dataset_name, root=dataset_root)
            torch.load = original_load
        else:
            raise e

    # Load the trained model
    model = load_trained_model(config, dataset.num_features, dataset.num_tasks, device)
    
    if model is None:
        print("No trained model found. Please train the model first.")
        return

    # Test on a few molecules
    print("\n" + "="*60)
    print("MOLECULE EFFECTIVENESS PREDICTIONS")
    print("="*60)

    for idx in test_indices:
        molecule = dataset[idx]
        result = predict_molecule(model, molecule, device, config)

        print(f"\nMolecule {idx}:")
        print(f"  HIV Active: {result['is_hiv_active']}")
        print(f"  Confidence: {result['probability']:.4f}")
        print(f"  Effectiveness Score: {result['effectiveness_score']:.2f}/100")
        print(f"  Category: {result['category']}")
        print(f"  Ground Truth: {bool(molecule.y.item())}")


def predict_batch(model, molecules, device, config):
    """
    Predict on a batch of molecules
    """
    results = []
    for molecule in molecules:
        result = predict_molecule(model, molecule, device, config)
        results.append(result)
    return results
