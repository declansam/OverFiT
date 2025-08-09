import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, GCNConv, GATConv, GINConv
from torch_geometric.data import Data
from rdkit import Chem
import warnings

warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ==============================================================================
# 1. SMILES-TO-GRAPH CONVERTER (CORRECTED VERSION)
# ==============================================================================

def smiles_to_pyg_data(smiles_string):
    """
    Converts a SMILES string to a PyTorch Geometric Data object with 9 features,
    matching the ogbg-molhiv dataset.
    """
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        raise ValueError(f"Invalid SMILES string provided: {smiles_string}")

    node_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),
            atom.GetChiralTag(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            int(atom.GetHybridization()),
            atom.GetImplicitValence(),
            atom.GetNumRadicalElectrons(),
            int(atom.GetIsAromatic()),
            int(atom.IsInRing())
        ]
        node_features.append(features)
    x = torch.tensor(node_features, dtype=torch.float)

    edge_indices = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i])

    if len(edge_indices) > 0:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index)


# ==============================================================================
# 3. PREDICTION FUNCTION
# ==============================================================================

def predict_on_new_graph(model, graph_data, device, threshold=0.5):
    """Predicts HIV activity for a single, new graph data object."""
    model.eval()
    with torch.no_grad():
        graph_data = graph_data.to(device)
        batch_vector = torch.zeros(graph_data.num_nodes, dtype=torch.long, device=device)
        out = model(graph_data.x, graph_data.edge_index, batch_vector)
        prob = torch.sigmoid(out).item()
        prediction = 1 if prob > threshold else 0
        effectiveness_score = prob * 100
        category = "Highly Effective" if effectiveness_score >= 80 else "Moderately Effective" if effectiveness_score >= 60 else "Weakly Effective"
    return {'is_hiv_active': bool(prediction), 'probability': prob, 'effectiveness_score': effectiveness_score, 'category': category}


if __name__ == "__main__":
    # --- MODEL LOADING ---
    MODEL_PARAMS = {'num_features': 9, 'num_classes': 1, 'hidden_dim': 256, 'dropout': 0.3}
    MODEL_PATH = 'best_hiv_gnn_model.pth'

    print("Instantiating and loading model...")
    model = HybridGNN(**MODEL_PARAMS).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Model weights loaded successfully from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {MODEL_PATH}. Please ensure it's in the correct directory.")
        exit()
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        exit()

    # --- PREDICTION FROM SMILES ---
    # Zidovudine (AZT) is an antiretroviral medication used to prevent and treat HIV/AIDS.
    # The model should ideally predict this as active.
    #TODO: Write the SMILES string here
    zidovudine_smiles = ''

    print(f"\nConverting SMILES string to graph: {zidovudine_smiles}")
    try:
        molecule_graph = smiles_to_pyg_data(zidovudine_smiles)
        print(f"Graph created successfully: {molecule_graph}")

        # Make a prediction
        result = predict_on_new_graph(model, molecule_graph, device)

        # Display the result
        print("\n" + "="*50)
        print("PREDICTION FOR ZIDOVUDINE (AZT)")
        print("="*50)
        print(f"  Predicted to be HIV Active: {result['is_hiv_active']}")
        print(f"  Confidence Probability:     {result['probability']:.4f}")
        print(f"  Effectiveness Score:        {result['effectiveness_score']:.2f} / 100")
        print(f"  Effectiveness Category:     '{result['category']}'")
        print("="*50)

    except ValueError as e:
        print(f"Error during graph conversion: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during prediction: {e}")