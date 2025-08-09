import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch_geometric.data import DataLoader
from torch_geometric.data.data import Data
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Fix for PyTorch 2.6+ compatibility with PyG datasets
import torch.serialization as ts
# Add safe globals for PyTorch Geometric data structures
ts.add_safe_globals([Data])
try:
    from torch_geometric.data.data import DataEdgeAttr
    ts.add_safe_globals([DataEdgeAttr])
except:
    pass

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class GINConvNet(nn.Module):
    """Graph Isomorphism Network for molecular property prediction"""

    def __init__(self, num_features, num_classes=1, hidden_dim=256, num_layers=5, dropout=0.2):
        super(GINConvNet, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # GIN convolution layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for layer in range(num_layers):
            if layer == 0:
                mlp = nn.Sequential(
                    nn.Linear(num_features, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            else:
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )

            conv = GINConv(mlp)
            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Readout layers - need num_layers + 1 to include initial features
        self.linear_predictions = nn.ModuleList()

        # Layer for initial features
        self.linear_predictions.append(nn.Linear(num_features, hidden_dim))

        # Layers for each GIN layer
        for layer in range(num_layers):
            self.linear_predictions.append(nn.Linear(hidden_dim, hidden_dim))

        # Final classifier
        self.final_dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        # Ensure input is float
        x = x.float()

        # Store hidden representations at each layer
        hidden_reps = [x]

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            hidden_reps.append(x)

        # Jumping knowledge connection - aggregate from all layers
        score_over_layer = 0
        for i, h in enumerate(hidden_reps):
            pooled = global_mean_pool(h, batch)
            score_over_layer += F.dropout(self.linear_predictions[i](pooled),
                                         self.dropout, training=self.training)

        # Final prediction
        x = self.final_dropout(score_over_layer)
        x = self.final_linear(x)
        x = F.relu(x)
        x = self.classifier(x)

        return x

class HybridGNN(nn.Module):
    """Hybrid GNN combining GCN, GAT, and GIN layers for robust molecular representations"""

    def __init__(self, num_features, num_classes=1, hidden_dim=256, dropout=0.3):
        super(HybridGNN, self).__init__()

        # Initial projection
        self.initial_proj = nn.Linear(num_features, hidden_dim)

        # GCN layers
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)

        # GAT layers - ensure output matches hidden_dim
        # With 4 heads and hidden_dim//4, concat results in hidden_dim
        self.gat1 = GATConv(hidden_dim, hidden_dim // 4, heads=4, concat=True, dropout=dropout)
        # Second GAT layer needs to account for concatenated input from first GAT
        self.gat2 = GATConv(hidden_dim, hidden_dim // 4, heads=4, concat=True, dropout=dropout)

        # GIN layer
        gin_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.gin = GINConv(gin_mlp)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.bn_gat = nn.BatchNorm1d(hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Final layers
        self.fc1 = nn.Linear(hidden_dim * 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x, edge_index, batch):
        # Ensure input is float
        x = x.float()

        # Initial projection
        x = self.initial_proj(x)
        x = F.relu(x)

        # GCN branch
        gcn_out = self.gcn1(x, edge_index)
        gcn_out = self.bn1(gcn_out)
        gcn_out = F.relu(gcn_out)
        gcn_out = self.dropout(gcn_out)
        gcn_out = self.gcn2(gcn_out, edge_index)

        # GAT branch
        gat_out = self.gat1(x, edge_index)
        gat_out = self.bn_gat(gat_out)  # Add batch norm after GAT
        gat_out = F.relu(gat_out)
        gat_out = self.dropout(gat_out)
        gat_out = self.gat2(gat_out, edge_index)

        # GIN branch
        gin_out = self.gin(x, edge_index)
        gin_out = self.bn2(gin_out)
        gin_out = F.relu(gin_out)
        gin_out = self.dropout(gin_out)

        # Global pooling for each branch
        gcn_pool = global_mean_pool(gcn_out, batch)
        gat_pool = global_mean_pool(gat_out, batch)
        gin_pool = global_mean_pool(gin_out, batch)

        # Concatenate all representations
        x = torch.cat([gcn_pool, gat_pool, gin_pool], dim=1)

        # Final classification layers
        x = self.fc1(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.classifier(x)

        return x

def load_dataset():
    """Load and prepare the OGB-molhiv dataset"""
    print("Loading OGB-molhiv dataset...")

    # Alternative approach if add_safe_globals doesn't work
    try:
        dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root='datasets/')
    except Exception as e:
        if "weights_only" in str(e):
            print("Attempting alternate loading method...")
            # Monkey patch torch.load temporarily
            original_load = torch.load
            torch.load = lambda *args, **kwargs: original_load(*args, **{k: v for k, v in kwargs.items() if k != 'weights_only'}, weights_only=False)
            dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root='datasets/')
            torch.load = original_load  # Restore original
        else:
            raise e

    split_idx = dataset.get_idx_split()
    train_idx = split_idx["train"]
    valid_idx = split_idx["valid"]
    test_idx = split_idx["test"]

    train_dataset = dataset[train_idx]
    valid_dataset = dataset[valid_idx]
    test_dataset = dataset[test_idx]

    print(f"Dataset loaded successfully!")
    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of features: {dataset.num_features}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")

    return dataset, train_dataset, valid_dataset, test_dataset

def train_epoch(model, device, loader, optimizer, criterion):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y.float())

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, device, loader, evaluator):
    """Evaluate model performance"""
    model.eval()
    y_true = []
    y_pred = []

    for batch in tqdm(loader, desc="Evaluating"):
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)

        y_true.append(batch.y.cpu())
        y_pred.append(out.cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    # Calculate ROC-AUC
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    result = evaluator.eval(input_dict)

    return result['rocauc']

def predict_molecule(model, molecule_data, device, threshold=0.5):
    """
    Predict HIV activity for a single molecule
    Returns: prediction (0/1), confidence score, effectiveness ranking
    """
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

def train_model(model_type='gin', epochs=100, batch_size=32, lr=0.001, weight_decay=0.0):
    """Main training function"""

    # Load dataset
    dataset, train_dataset, valid_dataset, test_dataset = load_dataset()

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    if model_type == 'gin':
        model = GINConvNet(
            num_features=dataset.num_features,
            num_classes=dataset.num_tasks,
            hidden_dim=256,
            num_layers=5,
            dropout=0.2
        ).to(device)
    else:  # hybrid
        model = HybridGNN(
            num_features=dataset.num_features,
            num_classes=dataset.num_tasks,
            hidden_dim=256,
            dropout=0.3
        ).to(device)

    print(f"\nModel architecture: {model_type.upper()}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )

    # Evaluator
    evaluator = Evaluator(name="ogbg-molhiv")

    # Training history
    train_losses = []
    valid_aucs = []
    test_aucs = []
    best_valid_auc = 0
    best_test_auc = 0

    print("\nStarting training...")
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        # Train
        train_loss = train_epoch(model, device, train_loader, optimizer, criterion)
        train_losses.append(train_loss)

        # Evaluate
        valid_auc = evaluate(model, device, valid_loader, evaluator)
        test_auc = evaluate(model, device, test_loader, evaluator)

        valid_aucs.append(valid_auc)
        test_aucs.append(test_auc)

        # Update learning rate
        scheduler.step(valid_auc)

        # Save best model
        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            best_test_auc = test_auc
            torch.save(model.state_dict(), 'best_hiv_gnn_model.pth')

        print(f"Loss: {train_loss:.4f} | Valid AUC: {valid_auc:.4f} | Test AUC: {test_auc:.4f}")
        print(f"Best Valid AUC: {best_valid_auc:.4f} | Best Test AUC: {best_test_auc:.4f}")

    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Over Time')
    ax1.grid(True)

    ax2.plot(valid_aucs, label='Validation AUC')
    ax2.plot(test_aucs, label='Test AUC')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('ROC-AUC')
    ax2.set_title('Model Performance Over Time')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

    print(f"\nTraining complete! Best Test AUC: {best_test_auc:.4f}")

    return model, best_test_auc

def demo_prediction():
    """Demo function to show how to use the trained model for prediction"""

    # Load the dataset to get a sample molecule
    print("Loading dataset for demo...")
    try:
        dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root='datasets/')
    except Exception as e:
        if "weights_only" in str(e):
            original_load = torch.load
            torch.load = lambda *args, **kwargs: original_load(*args, **{k: v for k, v in kwargs.items() if k != 'weights_only'}, weights_only=False)
            dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root='datasets/')
            torch.load = original_load
        else:
            raise e

    # Initialize model (must match the architecture used in training)
    model = GINConvNet(
        num_features=dataset.num_features,
        num_classes=dataset.num_tasks,
        hidden_dim=256,
        num_layers=5,
        dropout=0.2
    ).to(device)

    # Load trained weights
    try:
        model.load_state_dict(torch.load('best_hiv_gnn_model.pth', map_location=device, weights_only=True))
        print("Model loaded successfully!")
    except:
        try:
            model.load_state_dict(torch.load('best_hiv_gnn_model.pth', map_location=device, weights_only=False))
            print("Model loaded successfully!")
        except:
            print("No trained model found. Please train the model first.")
            return

    # Test on a few molecules
    print("\n" + "="*60)
    print("MOLECULE EFFECTIVENESS PREDICTIONS")
    print("="*60)

    test_indices = [0, 100, 200, 300, 400]  # Sample indices

    for idx in test_indices:
        molecule = dataset[idx]
        result = predict_molecule(model, molecule, device)

        print(f"\nMolecule {idx}:")
        print(f"  HIV Active: {result['is_hiv_active']}")
        print(f"  Confidence: {result['probability']:.4f}")
        print(f"  Effectiveness Score: {result['effectiveness_score']:.2f}/100")
        print(f"  Category: {result['category']}")
        print(f"  Ground Truth: {bool(molecule.y.item())}")

if __name__ == "__main__":
    # Train the model
    model, test_auc = train_model(
        model_type='hybrid',
        epochs=1,
        batch_size=32,
        lr=0.001,
        weight_decay=0.0
    )

    # Demo predictions
    demo_prediction()
