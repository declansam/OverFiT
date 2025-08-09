"""
GNN Model Architectures for Molecular Property Prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn import GCNConv, GATConv, GINConv


class GINConvNet(nn.Module):
    """
    Architecture 1:

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
    Architecture 2:

    Hybrid GNN combining GCN, GAT, and GIN layers for robust molecular representations
    """

    def __init__(self, num_features, num_classes=1, hidden_dim=256, dropout=0.3):
        super(HybridGNN, self).__init__()

        # Initial projection
        self.initial_proj = nn.Linear(num_features, hidden_dim)

        # GCN layers
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)

        ## First GAT layer with 4 heads and hidden_dim//4, concat results in hidden_dim
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


def get_model(config, num_features, num_classes, device):
    """
    Factory function to create and return a model

    Args:
        config: Configuration object containing model parameters.
        num_features (int): The number of input features.
        num_classes (int): The number of output classes.
        device (torch.device): The device to run the model on (CPU or GPU).

    Returns:
        nn.Module: The created model.
    """
    
    model_type = config.get('model.type')
    hidden_dim = config.get('model.hidden_dim', 256)
    dropout = config.get('model.dropout', 0.3)
    num_layers = config.get('model.num_layers', 5)

    if model_type == 'gin':
        model = GINConvNet(
            num_features=num_features,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
    elif model_type == 'hybrid':
        model = HybridGNN(
            num_features=num_features,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(device)
