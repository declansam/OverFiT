import torch
import torch.serialization as ts
from torch_geometric.data import DataLoader
from torch_geometric.data.data import Data
from ogb.graphproppred import PygGraphPropPredDataset
import warnings

warnings.filterwarnings('ignore')

# Fix for PyTorch 2.6+ compatibility with PyG datasets
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


def load_dataset(config_manager=None):
    """
    Load and prepare the OGB-molhiv dataset
    """

    # Get dataset configuration
    if config_manager:
        dataset_name = config_manager.config.get('dataset', {}).get('name', 'ogbg-molhiv')
        dataset_root = config_manager.config.get('dataset', {}).get('root', 'datasets')
    else:
        dataset_name = 'ogbg-molhiv'
        dataset_root = 'datasets'

    print(f"Loading {dataset_name} dataset...")

    # Alternative approach if add_safe_globals doesn't work
    try:
        dataset = PygGraphPropPredDataset(name=dataset_name, root=dataset_root)
    except Exception as e:
        if "weights_only" in str(e):
            print("Attempting alternate loading method...")
            original_load = torch.load
            torch.load = lambda *args, **kwargs: original_load(*args, **{k: v for k, v in kwargs.items() if k != 'weights_only'}, weights_only=False)
            dataset = PygGraphPropPredDataset(name=dataset_name, root=dataset_root)
            torch.load = original_load
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


def create_data_loaders(train_dataset, valid_dataset, test_dataset, batch_size=32):
    """
    Create data loaders for training, validation, and test sets
    """

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader
