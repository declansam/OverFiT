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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
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

class MolecularFeatureExtractor:
    """Extract molecular features for traditional ML models"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def extract_node_features(self, data):
        """Extract aggregated node features from molecular graph"""
        x = data.x.float()
        
        # Basic statistical features
        features = []
        
        # Node feature statistics
        features.extend([
            x.mean(dim=0).numpy(),  # Mean of each feature
            x.std(dim=0).numpy(),   # Std of each feature
            x.min(dim=0)[0].numpy(), # Min of each feature
            x.max(dim=0)[0].numpy(), # Max of each feature
            x.sum(dim=0).numpy(),   # Sum of each feature
        ])
        
        # Graph-level features
        num_nodes = x.size(0)
        features.extend([
            [num_nodes],  # Number of atoms
            [x.mean().item()],  # Overall mean
            [x.std().item()],   # Overall std
        ])
        
        return np.concatenate(features)
    
    def extract_edge_features(self, data):
        """Extract edge-based features"""
        edge_index = data.edge_index
        num_nodes = data.x.size(0)
        num_edges = edge_index.size(1)
        
        # Basic edge statistics
        features = [
            num_edges,  # Number of bonds
            num_edges / num_nodes if num_nodes > 0 else 0,  # Average degree
        ]
        
        # Degree distribution
        degrees = torch.bincount(edge_index[0], minlength=num_nodes).float()
        features.extend([
            degrees.mean().item(),
            degrees.std().item(),
            degrees.max().item(),
            degrees.min().item(),
        ])
        
        return np.array(features)
    
    def extract_structural_features(self, data):
        """Extract structural molecular features"""
        x = data.x.float()
        edge_index = data.edge_index
        num_nodes = x.size(0)
        
        if num_nodes == 0:
            return np.zeros(10)
        
        # Create adjacency matrix
        adj = torch.zeros(num_nodes, num_nodes)
        adj[edge_index[0], edge_index[1]] = 1
        
        features = []
        
        # Connectivity features
        features.append(torch.sum(adj).item())  # Total connections
        
        # Ring detection (simple cycle counting)
        # Count triangles (3-cycles)
        adj_squared = torch.mm(adj, adj)
        triangles = torch.trace(torch.mm(adj_squared, adj)) / 6
        features.append(triangles.item())
        
        # Path lengths (approximate)
        features.append(torch.sum(adj_squared).item())  # 2-hop connections
        
        # Clustering coefficient approximation
        degrees = torch.sum(adj, dim=1)
        possible_triangles = degrees * (degrees - 1) / 2
        clustering = torch.sum(triangles) / torch.sum(possible_triangles).clamp(min=1)
        features.append(clustering.item())
        
        # Add more structural features
        features.extend([
            torch.sum(degrees == 1).item(),  # Terminal nodes
            torch.sum(degrees == 2).item(),  # Chain nodes
            torch.sum(degrees >= 3).item(),  # Branch nodes
            torch.max(degrees).item(),       # Max degree
            torch.std(degrees).item(),       # Degree variance
            num_nodes,                       # Molecular size
        ])
        
        return np.array(features)
    
    def extract_features(self, data):
        """Extract comprehensive molecular features"""
        node_features = self.extract_node_features(data)
        edge_features = self.extract_edge_features(data)
        structural_features = self.extract_structural_features(data)
        
        # Combine all features
        combined_features = np.concatenate([
            node_features,
            edge_features,
            structural_features
        ])
        
        return combined_features
    
    def fit_transform(self, dataset):
        """Extract and normalize features for entire dataset"""
        features_list = []
        
        print("Extracting molecular features...")
        for data in tqdm(dataset):
            features = self.extract_features(data)
            features_list.append(features)
        
        features_array = np.array(features_list)
        
        # Handle any NaN or inf values
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Normalize features
        normalized_features = self.scaler.fit_transform(features_array)
        self.is_fitted = True
        
        return normalized_features
    
    def transform(self, dataset):
        """Transform features for new data"""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit_transform first.")
        
        features_list = []
        for data in dataset:
            features = self.extract_features(data)
            features_list.append(features)
        
        features_array = np.array(features_list)
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return self.scaler.transform(features_array)

class DeepMLP(nn.Module):
    """Deep Multi-Layer Perceptron for molecular property prediction"""
    
    def __init__(self, input_dim, num_classes=1, hidden_dims=[512, 256, 128, 64], dropout=0.3):
        super(DeepMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final classifier
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class TraditionalMLEnsemble:
    """Ensemble of traditional ML models for molecular property prediction"""
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200, 
                max_depth=15, 
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ),
            'logistic_regression': Pipeline([
                ('scaler', StandardScaler()),
                ('lr', LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    random_state=42
                ))
            ]),
            'svm': Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(
                    C=1.0,
                    kernel='rbf',
                    probability=True,
                    random_state=42
                ))
            ])
        }
        
        self.is_fitted = False
        self.feature_importance = {}
    
    def fit(self, X, y):
        """Train all models in the ensemble"""
        print("Training traditional ML models...")
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X, y.ravel())
            
            # Store feature importance if available
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = model.feature_importances_
            elif hasattr(model, 'named_steps') and hasattr(model.named_steps.get('lr', model.named_steps.get('svm')), 'coef_'):
                self.feature_importance[name] = np.abs(model.named_steps.get('lr', model.named_steps.get('svm')).coef_[0])
        
        self.is_fitted = True
    
    def predict_proba(self, X):
        """Get ensemble predictions"""
        if not self.is_fitted:
            raise ValueError("Models not fitted. Call fit first.")
        
        predictions = []
        for name, model in self.models.items():
            pred = model.predict_proba(X)[:, 1]  # Get positive class probability
            predictions.append(pred)
        
        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred
    
    def predict(self, X, threshold=0.5):
        """Get binary predictions"""
        proba = self.predict_proba(X)
        return (proba > threshold).astype(int)
    
    def evaluate(self, X, y):
        """Evaluate ensemble performance"""
        y_pred_proba = self.predict_proba(X)
        auc_score = roc_auc_score(y, y_pred_proba)
        
        # Individual model performance
        individual_scores = {}
        for name, model in self.models.items():
            pred = model.predict_proba(X)[:, 1]
            individual_scores[name] = roc_auc_score(y, pred)
        
        return auc_score, individual_scores

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

def train_traditional_ml_models():
    """Train traditional ML models on molecular features"""
    
    # Load dataset
    dataset, train_dataset, valid_dataset, test_dataset = load_dataset()
    
    # Extract features
    feature_extractor = MolecularFeatureExtractor()
    
    print("Extracting features for training set...")
    X_train = feature_extractor.fit_transform(train_dataset)
    y_train = np.array([data.y.item() for data in train_dataset])
    
    print("Extracting features for validation set...")
    X_valid = feature_extractor.transform(valid_dataset)
    y_valid = np.array([data.y.item() for data in valid_dataset])
    
    print("Extracting features for test set...")
    X_test = feature_extractor.transform(test_dataset)
    y_test = np.array([data.y.item() for data in test_dataset])
    
    print(f"Feature dimensions: {X_train.shape[1]}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_valid.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    
    # Train ensemble
    ensemble = TraditionalMLEnsemble()
    ensemble.fit(X_train, y_train)
    
    # Evaluate on validation set
    valid_auc, valid_individual = ensemble.evaluate(X_valid, y_valid)
    print(f"\nValidation Results:")
    print(f"Ensemble AUC: {valid_auc:.4f}")
    for name, score in valid_individual.items():
        print(f"{name}: {score:.4f}")
    
    # Evaluate on test set
    test_auc, test_individual = ensemble.evaluate(X_test, y_test)
    print(f"\nTest Results:")
    print(f"Ensemble AUC: {test_auc:.4f}")
    for name, score in test_individual.items():
        print(f"{name}: {score:.4f}")
    
    # Plot feature importance for tree-based models
    plt.figure(figsize=(15, 8))
    
    # Plot for models that have feature importance
    models_with_importance = {k: v for k, v in ensemble.feature_importance.items() 
                            if len(v) > 0}
    
    if models_with_importance:
        n_models = len(models_with_importance)
        fig, axes = plt.subplots(1, min(n_models, 3), figsize=(15, 5))
        if n_models == 1:
            axes = [axes]
        
        for idx, (name, importance) in enumerate(list(models_with_importance.items())[:3]):
            if idx < len(axes):
                # Show top 20 features
                top_indices = np.argsort(importance)[-20:]
                axes[idx].barh(range(20), importance[top_indices])
                axes[idx].set_title(f'Top Features - {name.replace("_", " ").title()}')
                axes[idx].set_xlabel('Feature Importance')
                axes[idx].set_ylabel('Feature Index')
        
        plt.tight_layout()
        plt.savefig('traditional_ml_feature_importance.png')
        plt.show()
    
    return ensemble, feature_extractor, test_auc

def train_deep_mlp():
    """Train deep MLP on molecular features"""
    
    # Load dataset
    dataset, train_dataset, valid_dataset, test_dataset = load_dataset()
    
    # Extract features
    feature_extractor = MolecularFeatureExtractor()
    
    print("Extracting features for MLP...")
    X_train = feature_extractor.fit_transform(train_dataset)
    y_train = np.array([data.y.item() for data in train_dataset])
    
    X_valid = feature_extractor.transform(valid_dataset)
    y_valid = np.array([data.y.item() for data in valid_dataset])
    
    X_test = feature_extractor.transform(test_dataset)
    y_test = np.array([data.y.item() for data in test_dataset])
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_valid_tensor = torch.FloatTensor(X_valid)
    y_valid_tensor = torch.FloatTensor(y_valid).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
    
    # Create data loaders
    train_dataset_mlp = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    valid_dataset_mlp = torch.utils.data.TensorDataset(X_valid_tensor, y_valid_tensor)
    test_dataset_mlp = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = torch.utils.data.DataLoader(train_dataset_mlp, batch_size=256, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset_mlp, batch_size=256, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset_mlp, batch_size=256, shuffle=False)
    
    # Initialize model
    input_dim = X_train.shape[1]
    model = DeepMLP(
        input_dim=input_dim,
        num_classes=1,
        hidden_dims=[512, 256, 128, 64],
        dropout=0.3
    ).to(device)
    
    print(f"MLP Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )
    
    # Training
    train_losses = []
    valid_aucs = []
    test_aucs = []
    best_valid_auc = 0
    best_test_auc = 0
    epochs = 2
    
    print("Training Deep MLP...")
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            # Validation
            valid_preds = []
            valid_targets = []
            for batch_x, batch_y in valid_loader:
                batch_x = batch_x.to(device)
                out = model(batch_x)
                valid_preds.append(torch.sigmoid(out).cpu())
                valid_targets.append(batch_y.cpu())
            
            valid_preds = torch.cat(valid_preds).numpy()
            valid_targets = torch.cat(valid_targets).numpy()
            valid_auc = roc_auc_score(valid_targets, valid_preds)
            valid_aucs.append(valid_auc)
            
            # Test
            test_preds = []
            test_targets = []
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                out = model(batch_x)
                test_preds.append(torch.sigmoid(out).cpu())
                test_targets.append(batch_y.cpu())
            
            test_preds = torch.cat(test_preds).numpy()
            test_targets = torch.cat(test_targets).numpy()
            test_auc = roc_auc_score(test_targets, test_preds)
            test_aucs.append(test_auc)
        
        scheduler.step(valid_auc)
        
        # Save best model
        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            best_test_auc = test_auc
            torch.save(model.state_dict(), 'best_mlp_model.pth')
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Valid AUC={valid_auc:.4f}, Test AUC={test_auc:.4f}")
    
    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('MLP Training Loss')
    ax1.grid(True)
    
    ax2.plot(valid_aucs, label='Validation AUC')
    ax2.plot(test_aucs, label='Test AUC')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('ROC-AUC')
    ax2.set_title('MLP Performance')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('mlp_training_history.png')
    plt.show()
    
    print(f"Best MLP Test AUC: {best_test_auc:.4f}")
    
    return model, feature_extractor, best_test_auc

def compare_all_models():
    """Compare GNN vs Traditional ML vs Deep MLP models"""
    
    print("="*60)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("="*60)
    
    results = {}
    
    # Train GNN model
    print("\n1. Training GNN Model...")
    try:
        gnn_model, gnn_auc = train_model(model_type='gin', epochs=2, batch_size=32)
        results['GNN (GIN)'] = gnn_auc
    except Exception as e:
        print(f"GNN training failed: {e}")
        results['GNN (GIN)'] = 0.0
    
    # Train Traditional ML
    print("\n2. Training Traditional ML Ensemble...")
    try:
        ml_ensemble, ml_extractor, ml_auc = train_traditional_ml_models()
        results['Traditional ML Ensemble'] = ml_auc
    except Exception as e:
        print(f"Traditional ML training failed: {e}")
        results['Traditional ML Ensemble'] = 0.0
    
    # Train Deep MLP
    print("\n3. Training Deep MLP...")
    try:
        mlp_model, mlp_extractor, mlp_auc = train_deep_mlp()
        results['Deep MLP'] = mlp_auc
    except Exception as e:
        print(f"MLP training failed: {e}")
        results['Deep MLP'] = 0.0
    
    # Summary comparison
    print("\n" + "="*60)
    print("FINAL RESULTS COMPARISON")
    print("="*60)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    for rank, (model_name, auc_score) in enumerate(sorted_results, 1):
        print(f"{rank}. {model_name}: {auc_score:.4f}")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    model_names = list(results.keys())
    auc_scores = list(results.values())
    
    bars = plt.bar(model_names, auc_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.ylabel('ROC-AUC Score')
    plt.title('Model Performance Comparison on HIV Dataset')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, score in zip(bars, auc_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{score:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()
    
    return results

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
    # Option 1: Run comprehensive comparison of all models
    print("Starting comprehensive model comparison...")
    results = compare_all_models()
    
    # Option 2: Train individual models (uncomment to use)
    """
    # Train GNN model only
    model, test_auc = train_model(
        model_type='hybrid',
        epochs=10,
        batch_size=32,
        lr=0.001,
        weight_decay=0.0
    )
    
    # Train Traditional ML only
    ensemble, feature_extractor, ml_auc = train_traditional_ml_models()
    
    # Train Deep MLP only
    mlp_model, mlp_extractor, mlp_auc = train_deep_mlp()
    """
    
    # Demo predictions
    demo_prediction()