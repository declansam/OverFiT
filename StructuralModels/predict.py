import torch
import numpy as np
from ogb.graphproppred import PygGraphPropPredDataset

from models import DeepMLP, MolecularFeatureExtractor, TraditionalMLEnsemble
from datasets import load_dataset
from config_manager import ConfigManager

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict_molecule_traditional_ml(ensemble, feature_extractor, molecule_data, threshold=0.5):
    """
    Predict HIV activity for a single molecule using Traditional ML
    """
    # Extract features for single molecule
    features = feature_extractor.extract_features(molecule_data)
    features = features.reshape(1, -1)  # Reshape for single prediction
    
    # Handle any NaN or inf values
    features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Normalize features using the fitted scaler
    features_normalized = feature_extractor.scaler.transform(features)
    
    # Get prediction probability
    prob = ensemble.predict_proba(features_normalized)[0]
    
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


def predict_molecule_mlp(model, feature_extractor, molecule_data, threshold=0.5):
    """
    Predict HIV activity for a single molecule using Deep MLP
    """
    model.eval()
    
    with torch.no_grad():
        
        # Extract features for single molecule
        features = feature_extractor.extract_features(molecule_data)
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Normalize features
        features_normalized = feature_extractor.scaler.transform(features.reshape(1, -1))
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features_normalized).to(device)
        
        # Get prediction
        out = model(features_tensor)
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


def demo_prediction(config_manager: ConfigManager):
    """
    Demo function to show how to use the trained models for prediction
    """
    
    # Load the dataset to get sample molecules
    print("Loading dataset for demo...")
    try:
        # Get dataset configuration from config_manager
        dataset_name = config_manager.config.get('dataset', {}).get('name', 'ogbg-molhiv')
        dataset_root = config_manager.config.get('dataset', {}).get('root', 'datasets')
        
        dataset = PygGraphPropPredDataset(name=dataset_name, root=dataset_root)
    except Exception as e:
        if "weights_only" in str(e):
            original_load = torch.load
            torch.load = lambda *args, **kwargs: original_load(*args, **{k: v for k, v in kwargs.items() if k != 'weights_only'}, weights_only=False)
            dataset = PygGraphPropPredDataset(name=dataset_name, root=dataset_root)
            torch.load = original_load
        else:
            raise e
    
    # Test indices
    test_indices = config_manager.config.get('demo', {}).get('test_indices', [0, 100, 200, 300, 400])
    
    print("\n" + "="*80)
    print("MOLECULE EFFECTIVENESS PREDICTIONS")
    print("="*80)
    
    # Try Traditional ML predictions
    print("\n--- Traditional ML Predictions ---")
    try:
        # Load traditional ML models
        ensemble_path = config_manager.get_model_path('traditional_ml_ensemble')
        extractor_path = config_manager.get_model_path('traditional_ml_feature_extractor')
        
        ensemble = TraditionalMLEnsemble.load(ensemble_path)
        feature_extractor = MolecularFeatureExtractor.load(extractor_path)
        
        for idx in test_indices:
            molecule = dataset[idx]
            result = predict_molecule_traditional_ml(ensemble, feature_extractor, molecule)
            
            print(f"\nMolecule {idx} (Traditional ML):")
            print(f"  HIV Active: {result['is_hiv_active']}")
            print(f"  Confidence: {result['probability']:.4f}")
            print(f"  Effectiveness Score: {result['effectiveness_score']:.2f}/100")
            print(f"  Category: {result['category']}")
            print(f"  Ground Truth: {bool(molecule.y.item())}")
    
    except Exception as e:
        print(f"Traditional ML prediction failed: {e}")
    
    # Try MLP predictions
    print("\n--- Deep MLP Predictions ---")
    try:
        # Load MLP model
        mlp_model = load_mlp_model_for_prediction(config_manager)
        mlp_extractor_path = config_manager.get_model_path('mlp_feature_extractor')
        mlp_feature_extractor = MolecularFeatureExtractor.load(mlp_extractor_path)
        
        for idx in test_indices:
            molecule = dataset[idx]
            result = predict_molecule_mlp(mlp_model, mlp_feature_extractor, molecule)
            
            print(f"\nMolecule {idx} (Deep MLP):")
            print(f"  HIV Active: {result['is_hiv_active']}")
            print(f"  Confidence: {result['probability']:.4f}")
            print(f"  Effectiveness Score: {result['effectiveness_score']:.2f}/100")
            print(f"  Category: {result['category']}")
            print(f"  Ground Truth: {bool(molecule.y.item())}")
    
    except Exception as e:
        print(f"MLP prediction failed: {e}")


def load_mlp_model_for_prediction(config_manager: ConfigManager, model_path=None):
    """
    Load a trained MLP model for prediction
    """
    
    # Get MLP configuration
    mlp_config = config_manager.get_training_config('mlp')
    
    # Determine input dimension (this should match the feature extraction)
    # For now, we'll use a standard dimension - in practice you'd want to save this info
    input_dim = 64  # This matches our MolecularFeatureExtractor output
    
    # Initialize model
    model = DeepMLP(
        input_dim=input_dim,
        num_classes=1,
        hidden_dims=mlp_config.get('hidden_dims', [512, 256, 128, 64]),
        dropout=mlp_config.get('dropout', 0.3)
    ).to(device)
    
    # Use provided path or get from config
    if model_path is None:
        model_path = config_manager.get_model_path('best_mlp_model')
    
    # Load trained weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"MLP model loaded successfully from {model_path}!")
    except:
        try:
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
            print(f"MLP model loaded successfully from {model_path}!")
        except Exception as e:
            print(f"Could not load MLP model from {model_path}. Error: {e}")
            return None
    
    return model


def batch_predict_traditional_ml(ensemble, feature_extractor, molecules, threshold=0.5):
    """
    Predict HIV activity for multiple molecules using Traditional ML
    """
    results = []
    for i, molecule in enumerate(molecules):
        result = predict_molecule_traditional_ml(ensemble, feature_extractor, molecule, threshold)
        result['molecule_index'] = i
        results.append(result)
    
    return results


def batch_predict_mlp(model, feature_extractor, molecules, threshold=0.5):
    """
    Predict HIV activity for multiple molecules using Deep MLP
    """
    if model is None:
        print("MLP model not loaded. Please load a model first.")
        return None
    
    results = []
    for i, molecule in enumerate(molecules):
        result = predict_molecule_mlp(model, feature_extractor, molecule, threshold)
        result['molecule_index'] = i
        results.append(result)
    
    return results
