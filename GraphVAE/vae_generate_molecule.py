"""
VAE Molecule Generation Script
Works with the VAE model
"""

import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Descriptors import ExactMolWt, MolLogP, NumHDonors, NumHAcceptors, TPSA
from rdkit.Chem.Lipinski import NumRotatableBonds
from rdkit.Chem import rdMolDescriptors
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

from vae_train import GraphVAE


class Generator:
    """Molecule generator"""
    
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = None
        self.hyperparameters = None
        self.load_model(model_path)
        
        # Atom vocabulary
        self.atom_vocab = {
            0: 'C', 1: 'N', 2: 'O', 3: 'S', 4: 'F', 5: 'Si', 6: 'P', 7: 'Cl', 
            8: 'Br', 9: 'Mg', 10: 'Na', 11: 'Ca', 12: 'Fe', 13: 'As', 14: 'Al', 
            15: 'I', 16: 'B', 17: 'V', 18: 'K', 19: 'Tl', 20: 'Yb', 21: 'Sb', 
            22: 'Sn', 23: 'Ag', 24: 'Pd', 25: 'Co', 26: 'Se', 27: 'Ti', 28: 'Zn', 
            29: 'H', 30: 'Li', 31: 'Ge', 32: 'Cu', 33: 'Au', 34: 'Ni', 35: 'Cd', 
            36: 'In', 37: 'Mn', 38: 'Zr', 39: 'Cr', 40: 'Pt', 41: 'Hg', 42: 'Pb'
        }
    
    def load_model(self, model_path):
        print(f"Loading model from {model_path}...")
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        except Exception as e:
            print(f"Error loading model: {e}")
            return
        
        self.hyperparameters = checkpoint['hyperparameters']
        
        self.model = GraphVAE(
            node_input_dim=self.hyperparameters['node_input_dim'],
            hidden_dim=self.hyperparameters['hidden_dim'],
            latent_dim=self.hyperparameters['latent_dim'],
            max_nodes=self.hyperparameters['max_nodes']
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print("Model loaded successfully!")
    
    def generate_molecules(self, num_samples=100, temperature=1.0, edge_threshold=0.5):
        print(f"Generating {num_samples} molecules...")
        
        generated_molecules = []
        generated_smiles = []
        valid_count = 0
        
        with torch.no_grad():
            for i in tqdm(range(0, num_samples, 16), desc="Generating"):
                batch_size = min(16, num_samples - i)
                
                # Sample latent vectors
                z = torch.randn(batch_size, self.hyperparameters['latent_dim'], device=self.device)
                z = z * temperature
                
                # Decode
                adj_logits, node_features = self.model.decode(z, self.hyperparameters['max_nodes'])
                
                for j in range(batch_size):
                    mol, smiles = self.create_molecule(adj_logits[j], node_features[j], edge_threshold)
                    generated_molecules.append(mol)
                    generated_smiles.append(smiles)
                    if mol is not None:
                        valid_count += 1
        
        validity_rate = valid_count / num_samples
        unique_smiles = set([s for s in generated_smiles if s])
        uniqueness_rate = len(unique_smiles) / max(1, len([s for s in generated_smiles if s]))
        
        print(f"Valid: {valid_count}/{num_samples} ({validity_rate:.2%})")
        print(f"Unique: {len(unique_smiles)} ({uniqueness_rate:.2%})")
        
        return generated_molecules, generated_smiles, validity_rate, uniqueness_rate
    
    def create_molecule(self, adj_logits, node_features, edge_threshold=0.5):
        """Create molecule from logits"""
        try:
            # Get atom types
            atom_indices = torch.argmax(node_features, dim=-1)
            
            # Get edges
            adj_probs = torch.sigmoid(adj_logits)
            
            # Create molecule
            mol = Chem.RWMol()
            atom_map = {}
            
            # Add atoms
            for i in range(min(20, self.hyperparameters['max_nodes'])):
                atom_idx = atom_indices[i].item()
                if atom_idx in self.atom_vocab:
                    atom_symbol = self.atom_vocab[atom_idx]
                    
                    # Skip some problematic atoms
                    if atom_symbol in ['H']:
                        continue
                    if atom_symbol in ['Li', 'Na', 'K'] and np.random.random() > 0.1:
                        continue
                    
                    try:
                        atom = Chem.Atom(atom_symbol)
                        mol_idx = mol.AddAtom(atom)
                        atom_map[i] = mol_idx
                    except:
                        continue
            
            if len(atom_map) < 3:
                return None, None
            
            # Add bonds
            atom_list = list(atom_map.keys())
            for i in range(len(atom_list)):
                for j in range(i + 1, len(atom_list)):
                    if adj_probs[atom_list[i], atom_list[j]] > edge_threshold:
                        try:
                            mol.AddBond(atom_map[atom_list[i]], atom_map[atom_list[j]], Chem.BondType.SINGLE)
                        except:
                            continue
            
            if mol.GetNumBonds() == 0:
                return None, None
            
            # Convert and sanitize
            mol = mol.GetMol()
            Chem.SanitizeMol(mol)
            smiles = Chem.MolToSmiles(mol)
            
            # Basic validation
            if len(smiles) > 3 and mol.GetNumAtoms() <= 25:
                return mol, smiles
            
        except:
            pass
        
        return None, None


def analyze_molecules(molecules):
    """Simple analysis"""
    valid_mols = [mol for mol in molecules if mol is not None]
    
    if not valid_mols:
        return None
    
    properties = []
    for mol in tqdm(valid_mols, desc="Analyzing"):
        try:
            props = {
                'MW': ExactMolWt(mol),
                'LogP': MolLogP(mol),
                'HBD': NumHDonors(mol),
                'HBA': NumHAcceptors(mol),
                'TPSA': TPSA(mol),
                'RotBonds': NumRotatableBonds(mol),
                'AromaticRings': rdMolDescriptors.CalcNumAromaticRings(mol),
                'NumAtoms': mol.GetNumAtoms(),
                'NumBonds': mol.GetNumBonds(),
                'NumHeteroatoms': rdMolDescriptors.CalcNumHeteroatoms(mol)
            }
            properties.append(props)
        except:
            continue
    
    if not properties:
        return None
    
    props_df = pd.DataFrame(properties)
    
    # Count drug-like molecules
    drug_like = 0
    for props in properties:
        violations = 0
        if props['MW'] > 500: violations += 1
        if props['LogP'] > 5: violations += 1
        if props['HBD'] > 5: violations += 1
        if props['HBA'] > 10: violations += 1
        if violations <= 1: drug_like += 1
    
    return {
        'valid_molecules': len(valid_mols),
        'drug_like': drug_like,
        'drug_like_rate': drug_like / len(properties),
        'properties_df': props_df
    }


def main():
    print("Starting molecule generation...")
    
    # Find model
    model_path = None
    for path in ['vae_best_model.pth', 'vae_final_model.pth']:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:  
        print("No model found. Please train first:")
        print("python vae_train.py")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate
    generator = Generator(model_path, device)
    molecules, smiles_list, validity_rate, uniqueness_rate = generator.generate_molecules(
        num_samples=500, temperature=1.0, edge_threshold=0.4
    )
    
    # Filter valid
    valid_molecules = [mol for mol in molecules if mol is not None]
    valid_smiles = [smiles for smiles in smiles_list if smiles is not None]
    
    print(f"\n=== Results ===")
    print(f"Valid molecules: {len(valid_molecules)}")
    print(f"Unique SMILES: {len(set(valid_smiles))}")
    
    # Save SMILES
    with open('vae_generated_smiles.txt', 'w') as f:
        for smiles in valid_smiles:
            f.write(f"{smiles}\n")
    print("SMILES saved to 'vae_generated_smiles.txt'")
    
    # Analyze
    analysis = analyze_molecules(valid_molecules)
    if analysis:
        print(f"\n=== Analysis ===")
        print(f"Drug-like molecules: {analysis['drug_like']}")
        print(f"Drug-like rate: {analysis['drug_like_rate']:.2%}")
        
        props_df = analysis['properties_df']
        print(f"\n=== Properties ===")
        print(props_df[['MW', 'LogP', 'HBD', 'HBA', 'NumHeteroatoms']].describe())
        
        props_df.to_csv('vae_molecule_properties.csv', index=False)
        print("Properties saved to 'vae_molecule_properties.csv'")
        
        # Show examples
        print(f"\n=== Examples ===")
        unique_examples = list(set(valid_smiles))[:10]
        for i, smiles in enumerate(unique_examples):
            print(f"{i+1:2d}: {smiles}")
    
    # Create visualization
    if len(valid_molecules) >= 9:
        print("\nCreating molecular visualization...")
        selected_mols = valid_molecules[:9]
        img = Draw.MolsToGridImage(selected_mols, molsPerRow=3, subImgSize=(250, 250))
        img.save('vae_molecules_grid.png')
        print("Visualization saved to 'vae_molecules_grid.png'")
    
    print("\nMolecule generation completed!")


if __name__ == "__main__":
    main()
