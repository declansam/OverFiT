"""
Dataset loading and processing functionality for molecular property prediction
"""
import os
import numpy as np
import pandas as pd
import random
import torch
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader, Dataset
from torch_geometric.data.data import Data
from torch_geometric.utils import from_smiles
import torch.serialization as ts
import warnings
from rdkit import Chem
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# Suppress RDKit warnings at the C++ level - must be set before any RDKit imports
os.environ['RDKIT_LOG_LEVEL'] = 'ERROR'
# Additional RDKit warning suppression
try:
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    pass

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='rdkit')
warnings.filterwarnings('ignore', message='.*DEPRECATION WARNING.*MorganGenerator.*')


class SMILESDataset(Dataset):
    """
    Custom dataset class for SMILES data that converts SMILES to PyG Data objects
    """
    def __init__(self, smiles_data):
        super(SMILESDataset, self).__init__()
        self.smiles_data = smiles_data
        
    def len(self):
        return len(self.smiles_data)
        
    def get(self, idx):
        smiles, label = self.smiles_data[idx]
        # Convert SMILES to PyG Data object (should work since we pre-filtered)
        data = from_smiles(smiles)
        if data is None:
            # Fallback to manual conversion if from_smiles fails
            data = smiles_to_pyg_data(smiles)
            
        # Should not be None at this point, but add safety check
        if data is None:
            raise ValueError(f"Failed to convert SMILES to graph: {smiles}")
        
        # Keep the label as a single value to match the expected format
        data.y = torch.tensor([label], dtype=torch.float)
        return data


def smiles_to_pyg_data(smiles_string):
    """
    Converts a SMILES string to a PyTorch Geometric Data object with 9 features,
    matching the ogbg-molhiv dataset.
    """
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        return None

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


def setup_pytorch_compatibility():
    """
    Setup PyTorch compatibility with PyG datasets
    """

    # Fix for PyTorch compatibility with PyG datasets
    try:
        # For PyTorch versions that support add_safe_globals
        if hasattr(ts, 'add_safe_globals'):
            ts.add_safe_globals([Data])
            try:
                from torch_geometric.data.data import DataEdgeAttr
                ts.add_safe_globals([DataEdgeAttr])
            except:
                pass
    except:
        pass


def load_dataset(config):
    """
    Load and prepare the dataset specified in config

    Args:
        config: Configuration object containing dataset parameters.

    Returns:
        dataset (PygGraphPropPredDataset): The loaded dataset.
        train_dataset, valid_dataset, test_dataset (torch_geometric.data.Dataset): The train, validation, and test splits.
    """
    
    dataset_name = config.get('dataset.name', 'ogbg-molhiv')
    root = config.get('dataset.root', 'datasets/')
    
    print(f"Loading {dataset_name} dataset...")
    
    # Setup compatibility first
    setup_pytorch_compatibility()

    # Alternative approach if add_safe_globals doesn't work
    try:
        dataset = PygGraphPropPredDataset(name=dataset_name, root=root)
    
    except Exception as e:

        # Handle specific error 
        if "weights_only" in str(e):
            print("Attempting alternate loading method...")
            original_load = torch.load
            torch.load = lambda *args, **kwargs: original_load(*args, **{k: v for k, v in kwargs.items() if k != 'weights_only'}, weights_only=False)
            dataset = PygGraphPropPredDataset(name=dataset_name, root=root)
            torch.load = original_load
        else:
            raise e

    smiles_path = os.path.join('./data', 'ogbg_molhiv', 'mapping', 'mol.csv.gz')
    if not os.path.exists(smiles_path):
        raise FileNotFoundError(f"SMILES file not found at {smiles_path}")

    smiles_df = pd.read_csv(smiles_path)
    smiles_x = smiles_df['smiles'].tolist()
    smiles_y = smiles_df['HIV_active'].tolist()
    
    smiles = []
    for i, s in enumerate(smiles_x):
        x = from_smiles(s)
        if x is not None:
            smiles.append((s, smiles_y[i]))
    
    print(f"Loaded {len(smiles)} valid SMILES from {len(smiles_x)} total")

    # Random shuffle SMILES and split into train/test/val
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(smiles))

    X, y = zip(*smiles)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create proper datasets using the custom SMILESDataset class
    train_dataset = SMILESDataset(list(zip(X_train, y_train)))
    test_dataset = SMILESDataset(list(zip(X_test, y_test)))


    # Logging
    # print(f"Dataset loaded successfully!")
    # print(f"Number of graphs: {len(dataset)}")
    # print(f"Number of features: {dataset.num_features}")
    # print(f"Number of classes: {dataset.num_classes}")
    # print(f"Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")

    # return dataset, train_dataset, valid_dataset, test_dataset
    return dataset, train_dataset, test_dataset


def create_data_loaders(train_dataset, test_dataset, config):
    """
    Create DataLoader objects for training and testing

    Args:
        train_dataset, test_dataset (list): The train and test splits. 
        config: Configuration object containing batch_size parameter.

    Returns:
        train_loader, test_loader (torch_geometric.data.DataLoader): The DataLoader objects for each dataset.
    """
    
    batch_size = config.get('training.batch_size', 32)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
