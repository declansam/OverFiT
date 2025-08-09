import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch_geometric.data import DataLoader
from torch_geometric.data.data import Data
from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer
from torch_geometric.explain.config import ExplanationType, ModelTaskLevel, ModelMode
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
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

class GINConvNetWithAttention(nn.Module):
    """GIN with attention weights for explainability"""

    def __init__(self, num_features, num_classes=1, hidden_dim=256, num_layers=5, dropout=0.2):
        super(GINConvNetWithAttention, self).__init__()

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

        # Attention mechanism for node importance
        self.node_attention = nn.ModuleList()
        for layer in range(num_layers):
            self.node_attention.append(nn.Linear(hidden_dim, 1))

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

        # Store attention weights for visualization
        self.attention_weights = []

    def forward(self, x, edge_index, batch, return_attention=False):
        # Ensure input is float
        x = x.float()

        # Store hidden representations and attention weights
        hidden_reps = [x]
        layer_attentions = []

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)

            # Calculate attention weights for this layer
            if return_attention:
                att_weights = torch.sigmoid(self.node_attention[i](x))
                layer_attentions.append(att_weights)
                x = x * att_weights  # Apply attention

            x = F.dropout(x, self.dropout, training=self.training)
            hidden_reps.append(x)

        # Store attention weights for later visualization
        if return_attention:
            self.attention_weights = layer_attentions

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

def load_trained_model(dataset, model_path='best_hiv_gnn_model.pth'):
    """Load a trained model, automatically detecting the model type"""

    try:
        # Load the state dict first to inspect it
        state_dict = torch.load(model_path, map_location=device, weights_only=False)

        # Detect model type based on keys in state dict
        if 'initial_proj.weight' in state_dict:
            # This is a HybridGNN model
            print("Detected HybridGNN model")
            model = HybridGNN(
                num_features=dataset.num_features,
                num_classes=dataset.num_tasks,
                hidden_dim=256,
                dropout=0.3
            ).to(device)
        elif 'convs.0.nn.0.weight' in state_dict:
            # This is a GINConvNetWithAttention model
            print("Detected GINConvNetWithAttention model")
            model = GINConvNetWithAttention(
                num_features=dataset.num_features,
                num_classes=dataset.num_tasks,
                hidden_dim=256,
                num_layers=5,
                dropout=0.2
            ).to(device)
        else:
            # Default to GINConvNetWithAttention
            print("Unknown model type, defaulting to GINConvNetWithAttention")
            model = GINConvNetWithAttention(
                num_features=dataset.num_features,
                num_classes=dataset.num_tasks,
                hidden_dim=256,
                num_layers=5,
                dropout=0.2
            ).to(device)

        # Load the state dict
        model.load_state_dict(state_dict)
        print(f"Successfully loaded trained model from {model_path}")
        return model

    except FileNotFoundError:
        print(f"Model file {model_path} not found. Using untrained model.")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

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

class MolecularExplainer:
    """Comprehensive explainability toolkit for molecular GNNs"""

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()

    def explain_with_gnnexplainer(self, molecule_data, target_class=0):
        """Use GNNExplainer to identify important subgraphs"""

        # Create explainer
        explainer = Explainer(
            model=self.model,
            algorithm=GNNExplainer(epochs=200, lr=0.01),
            explanation_type=ExplanationType.model,
            model_config=dict(
                mode=ModelMode.binary_classification,
                task_level=ModelTaskLevel.graph,
                return_type='log_probs'
            ),
            node_mask_type='attributes',
            edge_mask_type='object'
        )

        # Get explanation
        molecule_data = molecule_data.to(self.device)
        batch = torch.zeros(molecule_data.x.size(0), dtype=torch.long, device=self.device)

        explanation = explainer(
            x=molecule_data.x.float(),
            edge_index=molecule_data.edge_index,
            batch=batch,
            target=torch.tensor([target_class], device=self.device)
        )

        return explanation

    def compute_gradient_attribution(self, molecule_data):
        """Compute gradient-based feature attribution"""

        molecule_data = molecule_data.to(self.device)
        molecule_data.x = molecule_data.x.float().requires_grad_(True)

        batch = torch.zeros(molecule_data.x.size(0), dtype=torch.long, device=self.device)

        # Forward pass
        output = self.model(molecule_data.x, molecule_data.edge_index, batch)

        # Backward pass
        self.model.zero_grad()
        output.backward()

        # Get gradients
        node_importance = molecule_data.x.grad.abs().sum(dim=1)

        return node_importance.detach().cpu().numpy()

    def integrated_gradients(self, molecule_data, steps=50):
        """Compute integrated gradients for feature attribution"""

        molecule_data = molecule_data.to(self.device)
        batch = torch.zeros(molecule_data.x.size(0), dtype=torch.long, device=self.device)

        # Explicitly cast to float to allow for gradient computation
        molecule_data.x = molecule_data.x.float()

        # Create baseline (zero features)
        baseline = torch.zeros_like(molecule_data.x)

        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, steps).to(self.device)

        # Ensure gradients are tracked for the initial tensor
        molecule_data.x.requires_grad_(True)

        accumulated_gradients = torch.zeros_like(molecule_data.x).to(self.device)

        for alpha in alphas:
            # Interpolate between baseline and input
            interpolated = baseline + alpha * molecule_data.x

            # We need to re-attach this to the graph each time
            interpolated.requires_grad_(True)

            # Forward pass with interpolated input
            output = self.model(interpolated, molecule_data.edge_index, batch)

            # Backward pass with a scalar output
            self.model.zero_grad()
            # Sum the output to make it a scalar for backward()
            output.sum().backward()

            # Accumulate gradients, check for None just in case
            if interpolated.grad is not None:
                accumulated_gradients += interpolated.grad.detach()

        # Compute integrated gradients
        integrated_grads = (molecule_data.x - baseline) * accumulated_gradients / steps

        # Sum across features to get node importance
        node_importance = integrated_grads.abs().sum(dim=1)

        return node_importance.detach().cpu().numpy()

    def get_attention_weights(self, molecule_data):
        """Extract attention weights from model (if available)"""

        if not hasattr(self.model, 'attention_weights'):
            print("Model doesn't have attention mechanism")
            return None

        molecule_data = molecule_data.to(self.device)
        batch = torch.zeros(molecule_data.x.size(0), dtype=torch.long, device=self.device)

        # Forward pass with attention
        _ = self.model(molecule_data.x, molecule_data.edge_index, batch, return_attention=True)

        # Get averaged attention across layers
        if self.model.attention_weights:
            attention = torch.stack(self.model.attention_weights).mean(dim=0).squeeze()
            return attention.detach().cpu().numpy()

        return None

    def subgraph_importance_sampling(self, molecule_data, num_samples=100):
        """Monte Carlo sampling to estimate subgraph importance"""

        molecule_data = molecule_data.to(self.device)
        batch = torch.zeros(molecule_data.x.size(0), dtype=torch.long, device=self.device)

        # Get original prediction
        with torch.no_grad():
            original_output = self.model(molecule_data.x.float(), molecule_data.edge_index, batch)
            original_prob = torch.sigmoid(original_output).item()

        num_nodes = molecule_data.x.size(0)
        node_importance = np.zeros(num_nodes)

        # Sample random subgraphs
        for _ in range(num_samples):
            # Randomly mask nodes
            mask = torch.rand(num_nodes) > 0.5
            masked_x = molecule_data.x.clone()
            masked_x[~mask] = 0  # Zero out masked nodes

            # Get prediction with masked input
            with torch.no_grad():
                masked_output = self.model(masked_x.float(), molecule_data.edge_index, batch)
                masked_prob = torch.sigmoid(masked_output).item()

            # Attribute importance based on prediction change
            importance = abs(original_prob - masked_prob)
            node_importance[mask.cpu().numpy()] += importance

        # Normalize by number of times each node was included
        node_importance /= num_samples

        return node_importance

def visualize_molecular_explanation(molecule_data, node_importance, edge_importance=None,
                                   title="Molecular Substructure Importance"):
    """Visualize important substructures in a molecule"""

    # Create NetworkX graph
    G = nx.Graph()
    edge_index = molecule_data.edge_index.cpu().numpy()

    for i in range(molecule_data.x.size(0)):
        G.add_node(i)

    for i in range(edge_index.shape[1]):
        G.add_edge(int(edge_index[0, i]), int(edge_index[1, i]))

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Graph visualization with node importance
    ax1 = axes[0]
    pos = nx.spring_layout(G, k=0.8, iterations=50)

    # Normalize importance scores for coloring
    norm = Normalize(vmin=node_importance.min(), vmax=node_importance.max())
    cmap = cm.Reds

    # Draw nodes with importance-based colors
    nx.draw_networkx_nodes(G, pos, ax=ax1,
                          node_color=node_importance,
                          node_size=300,
                          cmap=cmap,
                          vmin=node_importance.min(),
                          vmax=node_importance.max())

    # Draw edges
    if edge_importance is not None:
        # Color edges by importance
        edges = G.edges()
        edge_colors = []
        for e in edges:
            idx = np.where((edge_index[0] == e[0]) & (edge_index[1] == e[1]))[0]
            if len(idx) > 0:
                edge_colors.append(edge_importance[idx[0]])
            else:
                edge_colors.append(0)

        nx.draw_networkx_edges(G, pos, ax=ax1,
                              edge_color=edge_colors,
                              edge_cmap=cm.Blues,
                              width=2)
    else:
        nx.draw_networkx_edges(G, pos, ax=ax1, edge_color='gray', width=1)

    nx.draw_networkx_labels(G, pos, ax=ax1, font_size=8)

    ax1.set_title('Graph Structure with Node Importance', fontsize=12, fontweight='bold')
    ax1.axis('off')

    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax1, fraction=0.046, pad=0.04)

    # 2. Node importance distribution
    ax2 = axes[1]
    ax2.bar(range(len(node_importance)), sorted(node_importance, reverse=True),
            color='#e74c3c', edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Node Rank', fontsize=11)
    ax2.set_ylabel('Importance Score', fontsize=11)
    ax2.set_title('Node Importance Distribution', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # 3. Heatmap of node features with importance
    ax3 = axes[2]

    # Weight features by importance
    weighted_features = molecule_data.x.cpu().detach().numpy() * node_importance.reshape(-1, 1)

    # Create heatmap
    im = ax3.imshow(weighted_features.T, aspect='auto', cmap='YlOrRd')
    ax3.set_xlabel('Node Index', fontsize=11)
    ax3.set_ylabel('Feature Index', fontsize=11)
    ax3.set_title('Feature Importance Heatmap', fontsize=12, fontweight='bold')

    # Add colorbar
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig

def explain_prediction(model, molecule_data, device, method='all'):
    """
    Comprehensive explanation of model prediction

    Args:
        model: Trained GNN model
        molecule_data: PyG Data object for a molecule
        device: torch device
        method: 'gradient', 'integrated', 'attention', 'sampling', 'gnnexplainer', or 'all'
    """

    explainer = MolecularExplainer(model, device)
    results = {}

    print("\n" + "="*60)
    print("MOLECULAR PREDICTION EXPLANATION")
    print("="*60)

    # Get prediction
    model.eval()
    with torch.no_grad():
        molecule_data = molecule_data.to(device)
        batch = torch.zeros(molecule_data.x.size(0), dtype=torch.long, device=device)
        output = model(molecule_data.x.float(), molecule_data.edge_index, batch)
        prob = torch.sigmoid(output).item()
        prediction = 1 if prob > 0.5 else 0

    print(f"Prediction: {'HIV-Active' if prediction else 'HIV-Inactive'}")
    print(f"Confidence: {prob:.4f}")
    print(f"Number of atoms: {molecule_data.x.size(0)}")
    print(f"Number of bonds: {molecule_data.edge_index.size(1) // 2}")

    # Apply different explanation methods
    if method in ['gradient', 'all']:
        print("\n1. Gradient Attribution...")
        grad_importance = explainer.compute_gradient_attribution(molecule_data)
        results['gradient'] = grad_importance

        # Visualize
        fig = visualize_molecular_explanation(
            molecule_data, grad_importance,
            title="Gradient-based Feature Attribution"
        )
        plt.savefig('explanation_gradient.png', dpi=300, bbox_inches='tight')
        plt.show()

    if method in ['integrated', 'all']:
        print("\n2. Integrated Gradients...")
        ig_importance = explainer.integrated_gradients(molecule_data)
        results['integrated_gradients'] = ig_importance

        # Visualize
        fig = visualize_molecular_explanation(
            molecule_data, ig_importance,
            title="Integrated Gradients Attribution"
        )
        plt.savefig('explanation_integrated_gradients.png', dpi=300, bbox_inches='tight')
        plt.show()

    if method in ['attention', 'all']:
        print("\n3. Attention Weights...")
        attention = explainer.get_attention_weights(molecule_data)
        if attention is not None:
            results['attention'] = attention

            # Visualize
            fig = visualize_molecular_explanation(
                molecule_data, attention,
                title="Attention-based Importance"
            )
            plt.savefig('explanation_attention.png', dpi=300, bbox_inches='tight')
            plt.show()

    if method in ['sampling', 'all']:
        print("\n4. Subgraph Importance Sampling...")
        sampling_importance = explainer.subgraph_importance_sampling(molecule_data)
        results['sampling'] = sampling_importance

        # Visualize
        fig = visualize_molecular_explanation(
            molecule_data, sampling_importance,
            title="Monte Carlo Subgraph Importance"
        )
        plt.savefig('explanation_sampling.png', dpi=300, bbox_inches='tight')
        plt.show()

    if method in ['gnnexplainer', 'all']:
        print("\n5. GNNExplainer...")
        try:
            explanation = explainer.explain_with_gnnexplainer(molecule_data, target_class=prediction)

            if hasattr(explanation, 'node_mask'):
                node_mask = explanation.node_mask.cpu().numpy()
                results['gnnexplainer_nodes'] = node_mask

                # Visualize
                fig = visualize_molecular_explanation(
                    molecule_data, node_mask,
                    title="GNNExplainer Node Importance"
                )
                plt.savefig('explanation_gnnexplainer.png', dpi=300, bbox_inches='tight')
                plt.show()
        except Exception as e:
            print(f"GNNExplainer failed: {e}")

    # Consensus importance
    if len(results) > 1:
        print("\n6. Creating Consensus Explanation...")

        # Normalize all importance scores to [0, 1]
        normalized_importances = []
        for key, importance in results.items():
            if importance is not None and len(importance) == molecule_data.x.size(0):
                norm_imp = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
                normalized_importances.append(norm_imp)

        if normalized_importances:
            # Average normalized importances
            consensus_importance = np.mean(normalized_importances, axis=0)

            # Visualize consensus
            fig = visualize_molecular_explanation(
                molecule_data, consensus_importance,
                title="Consensus Molecular Importance (All Methods)"
            )
            plt.savefig('explanation_consensus.png', dpi=300, bbox_inches='tight')
            plt.show()

            # Identify critical atoms
            threshold = np.percentile(consensus_importance, 75)
            critical_atoms = np.where(consensus_importance > threshold)[0]
            print(f"\nCritical atoms (top 25%): {critical_atoms.tolist()}")

            results['consensus'] = consensus_importance

    return results

def analyze_dataset_patterns(model, dataset, device, num_samples=100):
    """Analyze common patterns in HIV-active molecules"""

    print("\n" + "="*60)
    print("ANALYZING COMMON PATTERNS IN HIV-ACTIVE MOLECULES")
    print("="*60)

    explainer = MolecularExplainer(model, device)

    active_importances = []
    inactive_importances = []

    # Collect importance scores for active and inactive molecules
    for i in tqdm(range(min(num_samples, len(dataset))), desc="Analyzing molecules"):
        molecule = dataset[i]

        # Get importance scores
        importance = explainer.integrated_gradients(molecule)

        if molecule.y.item() == 1:
            active_importances.append(importance)
        else:
            inactive_importances.append(importance)

    # Statistical analysis
    if active_importances and inactive_importances:
        # Average importance patterns
        avg_active = np.mean([imp.mean() for imp in active_importances])
        avg_inactive = np.mean([imp.mean() for imp in inactive_importances])

        print(f"\nAverage node importance:")
        print(f"  HIV-Active molecules: {avg_active:.4f}")
        print(f"  HIV-Inactive molecules: {avg_inactive:.4f}")
        print(f"  Difference: {avg_active - avg_inactive:.4f}")

        # Variance in importance
        var_active = np.mean([imp.var() for imp in active_importances])
        var_inactive = np.mean([imp.var() for imp in inactive_importances])

        print(f"\nVariance in node importance:")
        print(f"  HIV-Active molecules: {var_active:.4f}")
        print(f"  HIV-Inactive molecules: {var_inactive:.4f}")

        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Distribution of mean importances
        ax1 = axes[0]
        ax1.hist([imp.mean() for imp in active_importances],
                bins=20, alpha=0.7, label='HIV-Active', color='#4ecdc4')
        ax1.hist([imp.mean() for imp in inactive_importances],
                bins=20, alpha=0.7, label='HIV-Inactive', color='#ff6b6b')
        ax1.set_xlabel('Mean Node Importance', fontsize=11)
        ax1.set_ylabel('Count', fontsize=11)
        ax1.set_title('Distribution of Mean Node Importance', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Distribution of max importances
        ax2 = axes[1]
        ax2.hist([imp.max() for imp in active_importances],
                bins=20, alpha=0.7, label='HIV-Active', color='#4ecdc4')
        ax2.hist([imp.max() for imp in inactive_importances],
                bins=20, alpha=0.7, label='HIV-Inactive', color='#ff6b6b')
        ax2.set_xlabel('Max Node Importance', fontsize=11)
        ax2.set_ylabel('Count', fontsize=11)
        ax2.set_title('Distribution of Max Node Importance', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig('pattern_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    return active_importances, inactive_importances

def predict_hiv_activity(model, molecule_data, device, threshold=0.5):
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
        model = GINConvNetWithAttention(
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

def demo_explainability():
    """Demo function showing comprehensive explainability analysis"""

    print("\n" + "="*80)
    print(" EXPLAINABILITY DEMO FOR HIV DRUG CLASSIFICATION ")
    print("="*80)

    # Load dataset
    dataset = load_dataset()[0]

    # Try to load a trained model
    model = load_trained_model(dataset)

    if model is None:
        # Fallback to untrained model for demo
        print("Using untrained model for demonstration")
        model = GINConvNetWithAttention(
            num_features=dataset.num_features,
            num_classes=dataset.num_tasks,
            hidden_dim=256,
            num_layers=5,
            dropout=0.2
        ).to(device)

    # Find interesting molecules to explain
    print("\nFinding interesting molecules to explain...")

    active_idx = None
    inactive_idx = None
    high_confidence_idx = None

    model.eval()
    with torch.no_grad():
        for i in range(min(500, len(dataset))):
            molecule = dataset[i].to(device)
            batch = torch.zeros(molecule.x.size(0), dtype=torch.long, device=device)
            output = model(molecule.x.float(), molecule.edge_index, batch)
            prob = torch.sigmoid(output).item()

            # Find a true positive (correctly predicted active)
            if molecule.y.item() == 1 and prob > 0.5 and active_idx is None:
                active_idx = i

            # Find a true negative (correctly predicted inactive)
            if molecule.y.item() == 0 and prob < 0.5 and inactive_idx is None:
                inactive_idx = i

            # Find a high confidence prediction
            if (prob > 0.9 or prob < 0.1) and high_confidence_idx is None:
                high_confidence_idx = i

            if all([active_idx, inactive_idx, high_confidence_idx]):
                break

    # Use default indices if not found
    active_idx = active_idx or 0
    inactive_idx = inactive_idx or 1
    high_confidence_idx = high_confidence_idx or 2

    print(f"\nSelected molecules for analysis:")
    print(f"  HIV-Active molecule: Index {active_idx}")
    print(f"  HIV-Inactive molecule: Index {inactive_idx}")
    print(f"  High confidence prediction: Index {high_confidence_idx}")

    # Explain each molecule
    molecules_to_explain = [
        (active_idx, "HIV-Active Molecule"),
        (inactive_idx, "HIV-Inactive Molecule"),
        (high_confidence_idx, "High Confidence Prediction")
    ]

    all_results = {}

    for idx, description in molecules_to_explain:
        print(f"\n{'='*60}")
        print(f"Analyzing: {description} (Index {idx})")
        print('='*60)

        molecule = dataset[idx]

        # Get comprehensive explanation
        results = explain_prediction(
            model, molecule, device,
            method='all'  # Use all explanation methods
        )

        all_results[idx] = results

        # Print top important atoms
        if 'consensus' in results:
            consensus = results['consensus']
            top_5_atoms = np.argsort(consensus)[-5:][::-1]
            print(f"\nTop 5 most important atoms: {top_5_atoms.tolist()}")
            print(f"Their importance scores: {consensus[top_5_atoms].round(3).tolist()}")

    # Analyze patterns across dataset
    print("\n" + "="*80)
    print(" DATASET-WIDE PATTERN ANALYSIS ")
    print("="*80)

    active_patterns, inactive_patterns = analyze_dataset_patterns(
        model, dataset, device, num_samples=50
    )

    # Create summary visualization
    create_explainability_summary(all_results, molecules_to_explain)

    print("\n" + "="*80)
    print(" EXPLAINABILITY ANALYSIS COMPLETE ")
    print("="*80)
    print("\nGenerated files:")
    print("  - explanation_gradient.png")
    print("  - explanation_integrated_gradients.png")
    print("  - explanation_attention.png")
    print("  - explanation_sampling.png")
    print("  - explanation_gnnexplainer.png")
    print("  - explanation_consensus.png")
    print("  - pattern_analysis.png")
    print("  - explainability_summary.png")

    return all_results

def create_explainability_summary(all_results, molecules_info):
    """Create a summary visualization of all explainability results"""

    num_molecules = len(all_results)
    num_methods = len(next(iter(all_results.values())))

    fig, axes = plt.subplots(num_molecules, 3, figsize=(15, 5*num_molecules))

    if num_molecules == 1:
        axes = axes.reshape(1, -1)

    for mol_idx, ((idx, description), ax_row) in enumerate(zip(molecules_info, axes)):
        results = all_results[idx]

        # Plot 1: Compare different methods
        ax1 = ax_row[0]
        method_names = []
        mean_importances = []

        for method, importance in results.items():
            if importance is not None and method != 'consensus':
                method_names.append(method.replace('_', ' ').title())
                mean_importances.append(np.mean(importance))

        ax1.bar(method_names, mean_importances, color=plt.cm.Set3(np.arange(len(method_names))))
        ax1.set_xlabel('Explanation Method', fontsize=10)
        ax1.set_ylabel('Mean Importance', fontsize=10)
        ax1.set_title(f'{description}: Method Comparison', fontsize=11, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)

        # Plot 2: Consensus importance distribution
        ax2 = ax_row[1]
        if 'consensus' in results:
            consensus = results['consensus']
            ax2.hist(consensus, bins=20, color='#3498db', edgecolor='black', alpha=0.7)
            ax2.axvline(consensus.mean(), color='red', linestyle='--', label=f'Mean: {consensus.mean():.3f}')
            ax2.set_xlabel('Importance Score', fontsize=10)
            ax2.set_ylabel('Number of Atoms', fontsize=10)
            ax2.set_title('Consensus Importance Distribution', fontsize=11, fontweight='bold')
            ax2.legend()
            ax2.grid(axis='y', alpha=0.3)

        # Plot 3: Top important atoms
        ax3 = ax_row[2]
        if 'consensus' in results:
            consensus = results['consensus']
            top_10_idx = np.argsort(consensus)[-10:][::-1]
            top_10_importance = consensus[top_10_idx]

            colors = plt.cm.Reds(top_10_importance / top_10_importance.max())
            bars = ax3.bar(range(10), top_10_importance, color=colors, edgecolor='black')
            ax3.set_xlabel('Atom Rank', fontsize=10)
            ax3.set_ylabel('Importance Score', fontsize=10)
            ax3.set_title('Top 10 Most Important Atoms', fontsize=11, fontweight='bold')
            ax3.set_xticks(range(10))
            ax3.set_xticklabels([f'#{i}' for i in top_10_idx], fontsize=8)
            ax3.grid(axis='y', alpha=0.3)

            # Add value labels on bars
            for bar, val in zip(bars, top_10_importance):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    plt.suptitle('Explainability Analysis Summary', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('explainability_summary.png', dpi=300, bbox_inches='tight')
    plt.show()

def identify_pharmacophores(model, dataset, device, num_samples=100):
    """Identify common important substructures (pharmacophores) in active molecules"""

    print("\n" + "="*60)
    print("IDENTIFYING PHARMACOPHORES")
    print("="*60)

    explainer = MolecularExplainer(model, device)

    # Collect subgraph patterns from active molecules
    active_subgraphs = []

    for i in tqdm(range(min(num_samples, len(dataset))), desc="Analyzing active molecules"):
        molecule = dataset[i]

        if molecule.y.item() == 1:  # Only HIV-active molecules
            # Get importance scores
            importance = explainer.integrated_gradients(molecule)

            # Identify important nodes (top 30%)
            threshold = np.percentile(importance, 70)
            important_nodes = np.where(importance > threshold)[0]

            if len(important_nodes) > 0:
                # Extract subgraph features
                subgraph_features = molecule.x[important_nodes].cpu().numpy()
                active_subgraphs.append({
                    'nodes': important_nodes,
                    'features': subgraph_features,
                    'importance': importance[important_nodes],
                    'size': len(important_nodes)
                })

    if active_subgraphs:
        # Analyze common patterns
        sizes = [sg['size'] for sg in active_subgraphs]
        avg_size = np.mean(sizes)

        print(f"\nPharmacophore Statistics:")
        print(f"  Number of active molecules analyzed: {len(active_subgraphs)}")
        print(f"  Average pharmacophore size: {avg_size:.1f} atoms")
        print(f"  Size range: {min(sizes)} - {max(sizes)} atoms")

        # Visualize pharmacophore statistics
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Size distribution
        ax1 = axes[0]
        ax1.hist(sizes, bins=15, color='#2ecc71', edgecolor='black', alpha=0.7)
        ax1.axvline(avg_size, color='red', linestyle='--', label=f'Mean: {avg_size:.1f}')
        ax1.set_xlabel('Pharmacophore Size (# atoms)', fontsize=11)
        ax1.set_ylabel('Count', fontsize=11)
        ax1.set_title('Distribution of Pharmacophore Sizes', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Average importance by position
        ax2 = axes[1]
        all_importances = []
        for sg in active_subgraphs:
            all_importances.extend(sg['importance'])

        ax2.hist(all_importances, bins=20, color='#e74c3c', edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Node Importance in Pharmacophore', fontsize=11)
        ax2.set_ylabel('Count', fontsize=11)
        ax2.set_title('Importance Distribution in Pharmacophores', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig('pharmacophore_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    return active_subgraphs

if __name__ == "__main__":
    # Option 1: Train the model
    print("Choose an option:")
    print("1. Train new model")
    print("2. Run explainability analysis on trained model")
    print("3. Both training and explainability")

    # For automated runs, default to option 3
    choice = 3  # Change this to 1, 2, or 3 as needed

    if choice in [1, 3]:
        # Train the model
        model, test_auc = train_model(
            model_type='hybrid',
            epochs=1,
            batch_size=32,
            lr=0.001,
            weight_decay=0.0
        )

    if choice in [2, 3]:
        # Run explainability analysis
        demo_explainability()

        # Optional: Identify pharmacophores
        print("\nWould you like to identify pharmacophores? (Resource intensive)")

        # Auto-run for demo
        dataset = load_dataset()[0]

        # Use the new load_trained_model function
        model = load_trained_model(dataset)

        if model is not None:
            pharmacophores = identify_pharmacophores(model, dataset, device, num_samples=50)
        else:
            print("Model not found. Train the model first for pharmacophore analysis.")
