import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, GCNConv, GATConv, GINConv
from torch_geometric.data import Data
from torch_geometric.utils import from_smiles
from rdkit import Chem
import warnings
from GNNModels.models import HybridGNN, GINConvNet

warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ==============================================================================
# PREDICTION FUNCTION
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


def main(smiles_string=None):

    # --- MODEL LOADING ---
    MODEL_PARAMS = {'num_features': 9, 'num_classes': 1, 'hidden_dim': 256, 'dropout': 0.3}
    MODEL_PATH = './GNNModels/models/best_hiv_gin_model_backup.pth'

    print("Instantiating and loading model...")
    model = GINConvNet(**MODEL_PARAMS).to(device)
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
    smiles_path = os.path.join('./data', 'ogbg_molhiv', 'mapping', 'mol.csv.gz')
    if not os.path.exists(smiles_path):
        raise FileNotFoundError(f"SMILES file not found at {smiles_path}")  
    smiles_df = pd.read_csv(smiles_path)

    # Find first entry where HIV_active is 1
    active_molecule = smiles_df[smiles_df['HIV_active'] == 1].iloc[0]
    print(f"Found active molecule at index {active_molecule.name} with SMILES: {active_molecule['smiles']}")

    test_smile = active_molecule['smiles']
    print(active_molecule['HIV_active'])
    
    # Use provided SMILES string if available, otherwise use the default active molecule
    if smiles_string:
        test_smile = smiles_string.strip()
    
    print(f"\nConverting SMILES string to graph: {test_smile}")
    
    try:
        molecule_graph = from_smiles(test_smile)
        print(f"Graph created successfully: {molecule_graph}")

        # Make a prediction
        result = predict_on_new_graph(model, molecule_graph, device)

        # Display the result
        print("\n" + "="*50)
        print("PREDICTION FOR ZIDOVUDINE (AZT)")
        print("="*50)
        print(f"  Predicted to be HIV Active: {result['is_hiv_active']}")
        print(f"  Confidence Probability:     {result['probability']:.4f}")
        print(f"  Effectiveness Score:        {result['effectiveness_score']:.2f}")
        print(f"  Effectiveness Category:     '{result['category']}'")
        print("="*50)

        return result

    except ValueError as e:
        print(f"Error during graph conversion: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during prediction: {e}")


if __name__ == "__main__":
    
    # Zidovudine (AZT) is an antiretroviral medication used to prevent and treat HIV/AIDS.
    # The model should ideally predict this as active.
    zidovudine_smiles = "C1CC1C#C[C@]2(C3=C(C=CC(=C3)Cl)NC(=O)O2)C(F)(F)F"
    main(zidovudine_smiles)
