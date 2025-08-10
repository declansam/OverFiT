import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn import GCNConv, GATConv, GINConv
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from tqdm import tqdm


class GINConvNet(nn.Module):
    """
    Graph Isomorphism Network for molecular property prediction
    """

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
    """
    Hybrid GNN combining GCN, GAT, and GIN layers for robust molecular representations
    """

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


class DeepMLP(nn.Module):
    """
    Deep Multi-Layer Perceptron for molecular property prediction
    """
    
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


class MolecularFeatureExtractor:
    """
    Extract molecular features for traditional ML models
    """
    
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
        """
        Extract edge-based features
        """
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
        """
        Extract structural molecular features
        """

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
        """
        Extract comprehensive molecular features
        """

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
        """
        Extract and normalize features for entire dataset
        """
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
        """
        Transform features for new data
        """

        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit_transform first.")
        
        features_list = []
        for data in dataset:
            features = self.extract_features(data)
            features_list.append(features)
        
        features_array = np.array(features_list)
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return self.scaler.transform(features_array)
    
    def save(self, filepath):
        """
        Save the feature extractor to a file
        """

        import joblib
        extractor_data = {
            'scaler': self.scaler,
            'is_fitted': self.is_fitted
        }
        joblib.dump(extractor_data, filepath)
        print(f"Feature extractor saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Load the feature extractor from a file
        """

        import joblib
        extractor_data = joblib.load(filepath)
        
        # Create a new instance
        instance = cls()
        instance.scaler = extractor_data['scaler']
        instance.is_fitted = extractor_data['is_fitted']
        
        print(f"Feature extractor loaded from: {filepath}")
        return instance


class TraditionalMLEnsemble:
    """
    Ensemble of traditional ML models for molecular property prediction
    """
    
    def __init__(self, config=None):
        
        # Default configuration
        default_config = {
            'random_forest': {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 5, 
                            'min_samples_leaf': 2, 'n_jobs': -1},
            'xgboost': {'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.1, 
                       'subsample': 0.8, 'colsample_bytree': 0.8, 'eval_metric': 'logloss'},
            'gradient_boosting': {'n_estimators': 150, 'max_depth': 6, 'learning_rate': 0.1, 'subsample': 0.8},
            'logistic_regression': {'C': 1.0, 'max_iter': 1000}
        }
        
        # Use provided config or defaults
        ml_config = config if config else default_config
        
        self.models = {
            'random_forest': RandomForestClassifier(
                random_state=42,
                **ml_config.get('random_forest', default_config['random_forest'])
            ),
            'xgboost': xgb.XGBClassifier(
                random_state=42,
                **ml_config.get('xgboost', default_config['xgboost'])
            ),
            'gradient_boosting': GradientBoostingClassifier(
                random_state=42,
                **ml_config.get('gradient_boosting', default_config['gradient_boosting'])
            ),
            'logistic_regression': Pipeline([
                ('scaler', StandardScaler()),
                ('lr', LogisticRegression(
                    random_state=42,
                    **ml_config.get('logistic_regression', default_config['logistic_regression'])
                ))
            ])
        }
        
        self.is_fitted = False
        self.feature_importance = {}
    
    def fit(self, X, y):
        """
        Train all models in the ensemble
        """

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
        """
        Get ensemble predictions
        """

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
        """
        Get binary predictions
        """

        proba = self.predict_proba(X)
        return (proba > threshold).astype(int)
    
    def evaluate(self, X, y):
        """
        Evaluate ensemble performance
        """

        y_pred_proba = self.predict_proba(X)
        auc_score = roc_auc_score(y, y_pred_proba)
        
        # Individual model performance
        individual_scores = {}
        for name, model in self.models.items():
            pred = model.predict_proba(X)[:, 1]
            individual_scores[name] = roc_auc_score(y, pred)
        
        return auc_score, individual_scores
    
    def save(self, filepath):
        """
        Save the ensemble models to a file
        """
        
        import joblib
        ensemble_data = {
            'models': self.models,
            'feature_importance': self.feature_importance,
            'is_fitted': self.is_fitted
        }
        joblib.dump(ensemble_data, filepath)
        print(f"Traditional ML ensemble saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Load the ensemble models from a file
        """
        
        import joblib
        ensemble_data = joblib.load(filepath)
        
        # Create a new instance
        instance = cls()
        instance.models = ensemble_data['models']
        instance.feature_importance = ensemble_data['feature_importance']
        instance.is_fitted = ensemble_data['is_fitted']
        
        print(f"Traditional ML ensemble loaded from: {filepath}")
        return instance
