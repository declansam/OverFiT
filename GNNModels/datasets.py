"""
Dataset loading and processing functionality for molecular property prediction
"""

import torch
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader
from torch_geometric.data.data import Data
import torch.serialization as ts
import warnings

warnings.filterwarnings('ignore')


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

    # Create data splits
    split_idx = dataset.get_idx_split()
    train_idx = split_idx["train"]
    valid_idx = split_idx["valid"]
    test_idx = split_idx["test"]

    train_dataset = dataset[train_idx]
    valid_dataset = dataset[valid_idx]
    test_dataset = dataset[test_idx]

    # Logging
    print(f"Dataset loaded successfully!")
    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of features: {dataset.num_features}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")

    return dataset, train_dataset, valid_dataset, test_dataset


def create_data_loaders(train_dataset, valid_dataset, test_dataset, config):
    """
    Create DataLoader objects for training, validation, and testing

    Args:
        train_dataset, valid_dataset, test_dataset (torch_geometric.data.Dataset): The train, validation, and test splits. 
        config: Configuration object containing batch_size parameter.

    Returns:
        train_loader, valid_loader, test_loader (torch_geometric.data.DataLoader): The DataLoader objects for each dataset.
    """
    
    batch_size = config.get('training.batch_size', 32)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader
