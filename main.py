import os
import torch
import torch.nn as nn
import numpy as np

# Fix for numpy compatibility issues with PyTorch model loading
import sys
try:
    import numpy._core.multiarray
except ImportError:
    if not hasattr(np, '_core'):
        np._core = np.core
    sys.modules['numpy._core.multiarray'] = sys.modules['numpy.core.multiarray']

from GNNModels.models import GINConvNet
from GNNModels.config_manager import Config
from GNNModels.datasets import load_dataset
from MorganFingerprintMLP.models import MLP as MorganFingerprintMLP
from MorganFingerprintMLP.data_loader import extract_features_from_smiles, prepare_data_loaders
from StructuralModels.models import TraditionalMLEnsemble, DeepMLP, MolecularFeatureExtractor
from Inference.gnn_predict import smiles_to_pyg_data

def smiles_to_graph(smiles):
    return smiles_to_pyg_data(smiles)

class StackedModel(nn.Module):

    def __init__(self, dataset):
        super().__init__()

        self.gnn_model = GINConvNet(
            num_features=dataset.num_features,
            num_classes=1                       # Use 1 for binary classification with sigmoid
        )
        self.morgan_mlp = MorganFingerprintMLP(
            input_dim=2072
        )
        self.traditional_ensemble = TraditionalMLEnsemble()
        self.deep_mlp = DeepMLP(
            input_dim=64
        )

        # Add feature extractor for structural models
        self.feature_extractor = MolecularFeatureExtractor()

        self.meta_classifier = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, input_data, model_type: str = "deep"):
        
        # Handle both SMILES strings and PyG Data objects
        if isinstance(input_data, str):
            
            # SMILES string input
            smiles = input_data
            graph = smiles_to_graph(smiles)
            morgan_features = extract_features_from_smiles(smiles)
            # Create batch tensor for single graph
            batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=graph.x.device)
        
        else:
            # PyG Data object input (from dataset)
            graph = input_data
            
            # For training, we can't extract Morgan features without SMILES
            morgan_features = torch.zeros(1, 2072, device=graph.x.device)
            
            # Create batch tensor for single graph
            batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=graph.x.device)

        with torch.no_grad():
            if isinstance(input_data, str):
                morgan_pred = self.morgan_mlp(morgan_features)
            else:
                morgan_pred = torch.zeros(1, 1, device=graph.x.device)
                
            gnn_pred = self.gnn_model(graph.x, graph.edge_index, batch)

            if model_type == "traditional":
                structure_pred = self.traditional_ensemble(graph)
            else:
                # Extract features for DeepMLP
                features = self.feature_extractor.extract_features(graph)
                features_tensor = torch.tensor(features, dtype=torch.float32, device=graph.x.device).unsqueeze(0)
                
                # Set model to eval mode for single sample inference
                training_mode = self.deep_mlp.training
                self.deep_mlp.eval()
                structure_pred = self.deep_mlp(features_tensor)
                self.deep_mlp.train(training_mode)              # Restore original mode

        # Ensure all predictions have the same shape (1, 1)
        gnn_pred = gnn_pred.view(1, 1)
        morgan_pred = morgan_pred.view(1, 1)
        structure_pred = structure_pred.view(1, 1)

        individual_preds = torch.cat([gnn_pred, morgan_pred, structure_pred], dim=1)  # Shape: (1, 3)
        meta_pred = self.meta_classifier(individual_preds)
        return meta_pred

    # train method
    def train_meta(self, epochs, optimizer, train_dataset):
        self.train()
        for epoch in range(epochs):
            print(f"Training epoch {epoch + 1}/{epochs}")
            total_loss = 0
            num_batches = 0
            
            for i, data in enumerate(train_dataset):
                if i >= 100:  # Limit training for demo purposes
                    break
                    
                optimizer.zero_grad()
                pred = self.forward(data)
                
                # Get the label from the data object
                target = data.y.float().view(-1, 1)
                
                loss = nn.BCELoss()(pred, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")


    @torch.no_grad()
    def predict(self, input_data, model_type: str = "deep"):
        
        # Handle both SMILES strings and PyG Data objects
        if isinstance(input_data, str):
            
            # SMILES string input
            smiles = input_data
            graph = smiles_to_graph(smiles)
            morgan_features = extract_features_from_smiles([smiles])  # Pass as list
            
            # Create batch tensor for single graph
            batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=graph.x.device)
        
        else:
            # PyG Data object input
            graph = input_data
            
            # For Data objects, we can't extract Morgan features without SMILES
            morgan_features = torch.zeros(1, 2072, device=graph.x.device)
            
            # Create batch tensor for single graph
            batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=graph.x.device)

        # Forward pass
        if isinstance(input_data, str):
            morgan_features_tensor = torch.tensor(morgan_features[0], dtype=torch.float32, device=graph.x.device).unsqueeze(0)  # Take first sample and add batch dim
            morgan_pred = self.morgan_mlp(morgan_features_tensor)
        else:
            # Skip Morgan prediction for graph data without SMILES
            morgan_pred = torch.zeros(1, 1, device=graph.x.device)
            
        gnn_pred = self.gnn_model(graph.x, graph.edge_index, batch)

        if model_type == "traditional":
            structure_pred = self.traditional_ensemble(graph)
        else:
            # Extract features for DeepMLP
            features = self.feature_extractor.extract_features(graph)
            features_tensor = torch.tensor(features, dtype=torch.float32, device=graph.x.device).unsqueeze(0)
            structure_pred = self.deep_mlp(features_tensor)

        # Ensure all predictions have the same shape (1, 1)
        gnn_pred = gnn_pred.view(1, 1)
        morgan_pred = morgan_pred.view(1, 1)
        structure_pred = structure_pred.view(1, 1)

        individual_preds = torch.cat([gnn_pred, morgan_pred, structure_pred], dim=1)  # Shape: (1, 3)
        meta_pred = self.meta_classifier(individual_preds)
        return meta_pred
        
    def load_models(self):
        
        # Load models with compatibility for older numpy versions
        try:
            # Load GIN model
            self.gnn_model.load_state_dict(torch.load("GNNModels/models/best_hiv_gin_model.pth", weights_only=False))
            
            # Load Morgan MLP model (check if it's a full checkpoint or just state_dict)
            morgan_checkpoint = torch.load("MorganFingerprintMLP/models/best_mlp_model_standard.pth", weights_only=False)
            if 'model_state_dict' in morgan_checkpoint:
                self.morgan_mlp.load_state_dict(morgan_checkpoint['model_state_dict'])
            else:
                self.morgan_mlp.load_state_dict(morgan_checkpoint)
            
            # Load Deep MLP model
            self.deep_mlp.load_state_dict(torch.load("StructuralModels/models/best_mlp_model.pth", weights_only=False))
            
        except Exception as e:
            print(f"Error loading models: {e}")
            
            # Print what's actually in the checkpoint for debugging
            if "MorganFingerprintMLP" in str(e):
                morgan_checkpoint = torch.load("MorganFingerprintMLP/models/best_mlp_model_standard.pth", weights_only=False)
                print(f"Morgan checkpoint keys: {list(morgan_checkpoint.keys())}")
                if 'model_state_dict' in morgan_checkpoint:
                    print(f"Model state dict keys: {list(morgan_checkpoint['model_state_dict'].keys())[:5]}...")
            raise e
    
    def save_models(self):

        torch.save(self.gnn_model.state_dict(), "GNNModels/gnn_model.pth")
        torch.save(self.morgan_mlp.state_dict(), "MorganFingerprintMLP/morgan_mlp.pth")
        torch.save(self.deep_mlp.state_dict(), "StructuralModels/structure_mlp.pth")
        torch.save(self.meta_classifier.state_dict(), "Complete/meta_classifier.pth")

def main():

    smiles = "C1CC1C#C[C@]2(C3=C(C=CC(=C3)Cl)NC(=O)O2)C(F)(F)F"

    # Get dataloader
    loader_config = Config("./GNNModels/config.yaml")
    dataset, train_dataset, val_dataset, test_dataset = load_dataset(loader_config)

    # Model
    model = StackedModel(
        dataset=dataset
    )
    model.load_models()

    # Train Meta
    model.train_meta(
        epochs=10, 
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001), 
        train_dataset=train_dataset
    )

    # Prediction
    model.eval()
    pred = model.predict(smiles)

    print(pred)

    model.save_models()

if __name__ == "__main__":
    main()
