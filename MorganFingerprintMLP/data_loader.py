"""
Data loading and feature extraction for ogb-molhiv dataset using SMILES and molecular fingerprints.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from ogb.graphproppred import PygGraphPropPredDataset
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Lipinski
import pickle
import os
from typing import Tuple, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class MolecularFeaturesDataset(Dataset):
    """Dataset class for molecular features extracted from SMILES."""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Initialize the dataset with features and labels.
        
        Args:
            features: Molecular features array of shape (n_samples, n_features)
            labels: Binary labels array of shape (n_samples,)
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels.flatten())
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def smiles_to_morgan_fingerprint(smiles: str, radius: int = 3, n_bits: int = 2048) -> np.ndarray:
    """
    Convert SMILES string to Morgan fingerprint.
    
    Args:
        smiles: SMILES string representation of molecule
        radius: Radius for Morgan fingerprint (default=3)
        n_bits: Number of bits in fingerprint (default=2048)
        
    Returns:
        Morgan fingerprint as numpy array, or zeros if invalid SMILES
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(n_bits)
        
        # Generate Morgan fingerprint using modern RDKit API
        try:
            # Use the new MorganGenerator (RDKit 2023.09+)
            from rdkit.Chem.rdMolDescriptors import GetMorganGenerator
            morgan_gen = GetMorganGenerator(radius=radius, fpSize=n_bits)
            fp = morgan_gen.GetFingerprintAsNumPy(mol)
            return fp
        except (ImportError, AttributeError):
            # Fallback to older API if MorganGenerator is not available
            # Suppress the deprecation warning since we already tried the new API
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=DeprecationWarning)
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            return np.array(fp)
        
    except:
        return np.zeros(n_bits)


def extract_molecular_descriptors(smiles: str) -> Dict[str, float]:
    """
    Extract various molecular descriptors from SMILES.
    
    Args:
        smiles: SMILES string representation of molecule
        
    Returns:
        Dictionary of molecular descriptors
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        
        descriptors = {
            # Basic descriptors
            'mol_weight': Descriptors.MolWt(mol),
            'logp': Crippen.MolLogP(mol),
            'tpsa': Descriptors.TPSA(mol),
            'hbd': Descriptors.NumHDonors(mol),
            'hba': Descriptors.NumHAcceptors(mol),
            
            # Structural descriptors
            'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'num_aromatic_rings': Descriptors.NumAromaticRings(mol),
            'num_saturated_rings': Descriptors.NumSaturatedRings(mol),
            'num_aliphatic_rings': Descriptors.NumAliphaticRings(mol),
            'ring_count': Descriptors.RingCount(mol),
            
            # Atom counts
            'heavy_atom_count': Descriptors.HeavyAtomCount(mol),
            'num_heteroatoms': Descriptors.NumHeteroatoms(mol),
            
            # Connectivity descriptors
            'bertz_ct': Descriptors.BertzCT(mol),
            'chi0v': Descriptors.Chi0v(mol),
            'chi1v': Descriptors.Chi1v(mol),
            'chi2v': Descriptors.Chi2v(mol),
            'chi3v': Descriptors.Chi3v(mol),
            'chi4v': Descriptors.Chi4v(mol),
            
            # Additional descriptors
            'fr_nh2': Descriptors.fr_NH2(mol),
            'fr_c_o': Descriptors.fr_C_O(mol),
            'fr_benzene': Descriptors.fr_benzene(mol),
            'slogp_vsb': Descriptors.SlogP_VSA2(mol),
            'smr_vsb': Descriptors.SMR_VSA1(mol),
            'peoe_vsb': Descriptors.PEOE_VSA1(mol),
        }
        
        return descriptors
    except:
        return {}


def load_ogb_molhiv_data(data_dir: str = "data") -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Load ogb-molhiv dataset and extract SMILES strings.
    
    Args:
        data_dir: Directory containing the dataset
        
    Returns:
        Tuple of (dataframe with SMILES and labels, split indices)
    """
    # Robust PyTorch compatibility handling for OGB dataset loading
    print("Loading OGB-MOLHIV dataset...")
    
    # Store original torch.load function
    original_load = torch.load
    
    def safe_torch_load(*args, **kwargs):
        """Patched torch.load that handles OGB compatibility issues."""
        # Force weights_only=False for OGB dataset files
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    
    # Temporarily patch torch.load
    torch.load = safe_torch_load
    
    try:
        # Load the dataset with patched torch.load
        dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root=data_dir)
        print(f"Dataset loaded successfully with {len(dataset)} molecules")
        
        # Get split indices
        split_idx = dataset.get_idx_split()
        
        # Load SMILES data
        smiles_path = os.path.join(data_dir, 'ogbg_molhiv', 'mapping', 'mol.csv.gz')
        if not os.path.exists(smiles_path):
            raise FileNotFoundError(f"SMILES file not found at {smiles_path}")
        
        smiles_df = pd.read_csv(smiles_path)
        print(f"SMILES data loaded: {len(smiles_df)} molecules")
        
        # Extract labels from dataset
        labels = []
        print("Extracting labels from dataset...")
        for i in range(len(dataset)):
            data = dataset[i]
            labels.append(data.y.item())
        
        print(f"   Labels extracted: {len(labels)} molecules")
        print(f"   HIV-active: {sum(labels)} ({100*sum(labels)/len(labels):.1f}%)")
        print(f"   HIV-inactive: {len(labels)-sum(labels)} ({100*(len(labels)-sum(labels))/len(labels):.1f}%)")
        
        # Verify data consistency
        if len(smiles_df) != len(labels):
            raise ValueError(f"Mismatch: {len(smiles_df)} SMILES vs {len(labels)} labels")
        
        # Create dataframe with SMILES and labels
        df = pd.DataFrame({
            'smiles': smiles_df['smiles'],
            'label': labels
        })
        
        return df, split_idx
        
    except Exception as e:
        print(f"Error loading OGB dataset: {e}")
        raise e
    finally:
        # Always restore original torch.load
        torch.load = original_load


def extract_features_from_smiles(smiles_list, 
                                morgan_radius: int = 3, 
                                morgan_bits: int = 2048,
                                include_descriptors: bool = True) -> np.ndarray:
    """
    Extract comprehensive features from SMILES strings.
    
    Args:
        smiles_list: List of SMILES strings
        morgan_radius: Radius for Morgan fingerprint
        morgan_bits: Number of bits in Morgan fingerprint
        include_descriptors: Whether to include molecular descriptors
        
    Returns:
        Feature matrix of shape (n_molecules, n_features)
    """
    features = []
    
    print("Extracting molecular features...")
    for i, smiles in enumerate(smiles_list):
        if i % 1000 == 0:
            print(f"Processing molecule {i}/{len(smiles_list)}")
        
        # Morgan fingerprint
        morgan_fp = smiles_to_morgan_fingerprint(smiles, morgan_radius, morgan_bits)
        
        molecule_features = morgan_fp.tolist()
        
        # Add molecular descriptors if requested
        if include_descriptors:
            descriptors = extract_molecular_descriptors(smiles)
            
            # Define expected descriptor names (in order)
            descriptor_names = [
                'mol_weight', 'logp', 'tpsa', 'hbd', 'hba',
                'num_rotatable_bonds', 'num_aromatic_rings', 'num_saturated_rings',
                'num_aliphatic_rings', 'ring_count', 'heavy_atom_count', 'num_heteroatoms',
                'bertz_ct', 'chi0v', 'chi1v', 'chi2v', 'chi3v', 'chi4v',
                'fr_nh2', 'fr_c_o', 'fr_benzene', 'slogp_vsb', 'smr_vsb', 'peoe_vsb'
            ]
            
            # Add descriptor values (use 0 if missing)
            for desc_name in descriptor_names:
                molecule_features.append(descriptors.get(desc_name, 0.0))
        
        features.append(molecule_features)
    
    return np.array(features)


def prepare_data_loaders(data_dir: str = "data",
                        morgan_radius: int = 3,
                        morgan_bits: int = 2048,
                        include_descriptors: bool = True,
                        batch_size: int = 128,
                        cache_features: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    """
    Prepare data loaders for training, validation, and testing.
    
    Args:
        data_dir: Directory containing the dataset
        morgan_radius: Radius for Morgan fingerprint
        morgan_bits: Number of bits in Morgan fingerprint
        include_descriptors: Whether to include molecular descriptors
        batch_size: Batch size for data loaders
        cache_features: Whether to cache extracted features
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, scaler)
    """
    cache_file = f"mlp/cached_features_r{morgan_radius}_b{morgan_bits}_desc{include_descriptors}.pkl"
    
    # Try to load cached features
    if cache_features and os.path.exists(cache_file):
        print("Loading cached features...")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
            features = cached_data['features']
            labels = cached_data['labels']
            split_idx = cached_data['split_idx']
    else:
        print("Extracting features from scratch...")
        # Load data
        df, split_idx = load_ogb_molhiv_data(data_dir)
        
        # Extract features
        features = extract_features_from_smiles(
            df['smiles'].tolist(),
            morgan_radius=morgan_radius,
            morgan_bits=morgan_bits,
            include_descriptors=include_descriptors
        )
        
        labels = df['label'].values
        
        # Cache features if requested
        if cache_features:
            print(f"Caching features to {cache_file}...")
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'features': features,
                    'labels': labels,
                    'split_idx': split_idx
                }, f)
    
    # Split data
    train_idx = split_idx['train']
    val_idx = split_idx['valid']
    test_idx = split_idx['test']
    
    X_train = features[train_idx]
    y_train = labels[train_idx]
    X_val = features[val_idx]
    y_val = labels[val_idx]
    X_test = features[test_idx]
    y_test = labels[test_idx]
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Create datasets
    train_dataset = MolecularFeaturesDataset(X_train_scaled, y_train)
    val_dataset = MolecularFeaturesDataset(X_val_scaled, y_val)
    test_dataset = MolecularFeaturesDataset(X_test_scaled, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Data loaded: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    print(f"Feature dimension: {features.shape[1]}")
    print(f"Class distribution - Train: {np.bincount(y_train)}, Val: {np.bincount(y_val)}, Test: {np.bincount(y_test)}")
    
    return train_loader, val_loader, test_loader, scaler


if __name__ == "__main__":
    # Test the data loading
    train_loader, val_loader, test_loader, scaler = prepare_data_loaders()
    
    # Print some statistics
    for batch_features, batch_labels in train_loader:
        print(f"Batch features shape: {batch_features.shape}")
        print(f"Batch labels shape: {batch_labels.shape}")
        print(f"Label distribution in batch: {torch.bincount(batch_labels)}")
        break
