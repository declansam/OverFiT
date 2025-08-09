import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ogb.graphproppred import PygGraphPropPredDataset
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, Lipinski, Crippen
from rdkit.Chem.Draw import IPythonConsole
from collections import Counter
import networkx as nx
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Fix for PyTorch 2.6+ compatibility
import torch.serialization as ts
from torch_geometric.data.data import Data
ts.add_safe_globals([Data])
try:
    from torch_geometric.data.data import DataEdgeAttr
    ts.add_safe_globals([DataEdgeAttr])
except:
    pass

def load_dataset_for_viz():
    """Load the OGB-molhiv dataset with error handling"""
    print("Loading OGB-molhiv dataset for visualization...")
    try:
        dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root='datasets/')
    except Exception as e:
        if "weights_only" in str(e):
            print("Using alternate loading method...")
            original_load = torch.load
            torch.load = lambda *args, **kwargs: original_load(*args, **{k: v for k, v in kwargs.items() if k != 'weights_only'}, weights_only=False)
            dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root='datasets/')
            torch.load = original_load
        else:
            raise e

    print(f"Dataset loaded: {len(dataset)} molecules")
    return dataset

def visualize_class_distribution(dataset):
    """Visualize the distribution of HIV-active vs HIV-inactive molecules"""

    # Get labels
    labels = []
    for data in tqdm(dataset, desc="Extracting labels"):
        labels.append(data.y.item())

    labels = np.array(labels)

    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Basic pie chart
    ax1 = axes[0, 0]
    counts = Counter(labels)
    colors = ['#ff6b6b', '#4ecdc4']
    wedges, texts, autotexts = ax1.pie(
        counts.values(),
        labels=['HIV-Inactive (0)', 'HIV-Active (1)'],
        colors=colors,
        autopct='%1.2f%%',
        startangle=90,
        explode=(0.05, 0)
    )
    ax1.set_title('Distribution of HIV Activity in Dataset', fontsize=14, fontweight='bold')

    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)

    # 2. Bar chart with counts
    ax2 = axes[0, 1]
    bar_colors = ['#ff6b6b', '#4ecdc4']
    bars = ax2.bar(['HIV-Inactive', 'HIV-Active'], counts.values(), color=bar_colors, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Number of Molecules', fontsize=12)
    ax2.set_title('Molecule Count by HIV Activity', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontweight='bold')

    # 3. Split distribution (train/valid/test)
    ax3 = axes[1, 0]
    split_idx = dataset.get_idx_split()
    split_data = []

    for split_name, indices in [('Train', split_idx['train']),
                                ('Valid', split_idx['valid']),
                                ('Test', split_idx['test'])]:
        split_labels = [dataset[i].y.item() for i in indices]
        split_counter = Counter(split_labels)
        split_data.append({
            'split': split_name,
            'inactive': split_counter[0],
            'active': split_counter[1],
            'total': len(split_labels)
        })

    x = np.arange(len(split_data))
    width = 0.35

    inactive_counts = [d['inactive'] for d in split_data]
    active_counts = [d['active'] for d in split_data]

    bars1 = ax3.bar(x - width/2, inactive_counts, width, label='HIV-Inactive', color='#ff6b6b', edgecolor='black')
    bars2 = ax3.bar(x + width/2, active_counts, width, label='HIV-Active', color='#4ecdc4', edgecolor='black')

    ax3.set_xlabel('Dataset Split', fontsize=12)
    ax3.set_ylabel('Number of Molecules', fontsize=12)
    ax3.set_title('Class Distribution Across Splits', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([d['split'] for d in split_data])
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom', fontsize=9)

    # 4. Imbalance ratio visualization
    ax4 = axes[1, 1]
    imbalance_data = []
    for d in split_data:
        ratio = d['active'] / d['inactive'] if d['inactive'] > 0 else 0
        imbalance_data.append({
            'split': d['split'],
            'ratio': ratio,
            'percentage_active': (d['active'] / d['total']) * 100
        })

    x_pos = np.arange(len(imbalance_data))
    bars = ax4.bar(x_pos, [d['percentage_active'] for d in imbalance_data],
                   color=['#3498db', '#e74c3c', '#2ecc71'], edgecolor='black', linewidth=1.5)

    ax4.set_xlabel('Dataset Split', fontsize=12)
    ax4.set_ylabel('HIV-Active Percentage (%)', fontsize=12)
    ax4.set_title('HIV-Active Percentage by Split', fontsize=14, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([d['split'] for d in imbalance_data])
    ax4.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% (Balanced)')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)

    # Add percentage labels
    for bar, d in zip(bars, imbalance_data):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{d["percentage_active"]:.1f}%',
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('hiv_dataset_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print statistics
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"Total molecules: {len(labels):,}")
    print(f"HIV-Inactive (0): {counts[0]:,} ({counts[0]/len(labels)*100:.2f}%)")
    print(f"HIV-Active (1): {counts[1]:,} ({counts[1]/len(labels)*100:.2f}%)")
    print(f"Imbalance ratio (Active/Inactive): {counts[1]/counts[0]:.4f}")
    print("\nSplit-wise distribution:")
    for d in split_data:
        print(f"  {d['split']}: {d['total']:,} molecules "
              f"(Inactive: {d['inactive']:,}, Active: {d['active']:,})")

def graph_to_smiles(data):
    """Convert a PyG graph to SMILES string"""
    # Node features to atomic numbers mapping (simplified)
    # The first feature in OGB-molhiv typically represents atom type
    atom_type_map = {
        0: 6,   # Carbon
        1: 7,   # Nitrogen
        2: 8,   # Oxygen
        3: 9,   # Fluorine
        4: 15,  # Phosphorus
        5: 16,  # Sulfur
        6: 17,  # Chlorine
        7: 35,  # Bromine
        8: 53,  # Iodine
    }

    # Create RDKit molecule
    mol = Chem.RWMol()

    # Add atoms
    atom_indices = []
    for i in range(data.x.shape[0]):
        atom_type = int(data.x[i, 0].item()) if data.x[i, 0] < len(atom_type_map) else 0
        atomic_num = atom_type_map.get(atom_type, 6)  # Default to Carbon
        atom = Chem.Atom(atomic_num)
        atom_indices.append(mol.AddAtom(atom))

    # Add bonds
    edge_index = data.edge_index.numpy()
    added_bonds = set()

    for i in range(edge_index.shape[1]):
        src, dst = int(edge_index[0, i]), int(edge_index[1, i])
        bond_tuple = tuple(sorted([src, dst]))

        if bond_tuple not in added_bonds and src < len(atom_indices) and dst < len(atom_indices):
            mol.AddBond(src, dst, Chem.BondType.SINGLE)
            added_bonds.add(bond_tuple)

    # Convert to SMILES
    try:
        Chem.SanitizeMol(mol)
        smiles = Chem.MolToSmiles(mol)
        return smiles
    except:
        return None

def visualize_molecules(dataset, indices=None, max_mols=6):
    """Visualize molecular structures from the dataset"""

    if indices is None:
        # Get a mix of active and inactive molecules
        active_indices = []
        inactive_indices = []

        for i in range(min(1000, len(dataset))):
            if dataset[i].y.item() == 1 and len(active_indices) < max_mols//2:
                active_indices.append(i)
            elif dataset[i].y.item() == 0 and len(inactive_indices) < max_mols//2:
                inactive_indices.append(i)

            if len(active_indices) >= max_mols//2 and len(inactive_indices) >= max_mols//2:
                break

        indices = active_indices + inactive_indices

    molecules = []
    labels = []
    smiles_list = []

    print("\nGenerating molecular structures...")
    for idx in tqdm(indices[:max_mols], desc="Processing molecules"):
        data = dataset[idx]
        smiles = graph_to_smiles(data)

        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                molecules.append(mol)
                labels.append(f"Mol {idx}\nHIV: {'Active' if data.y.item() == 1 else 'Inactive'}")
                smiles_list.append(smiles)

    if molecules:
        # Create molecular structure visualization
        img = Draw.MolsToGridImage(
            molecules,
            molsPerRow=3,
            subImgSize=(300, 300),
            legends=labels,
            returnPNG=False
        )

        # Display the image
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.imshow(img)
        ax.axis('off')
        ax.set_title('Sample Molecular Structures from OGB-molhiv Dataset',
                    fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('molecular_structures.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Print SMILES
        print("\n" + "="*60)
        print("SMILES REPRESENTATIONS")
        print("="*60)
        for i, (idx, smiles) in enumerate(zip(indices[:len(smiles_list)], smiles_list)):
            activity = "Active" if dataset[idx].y.item() == 1 else "Inactive"
            print(f"Molecule {idx} (HIV-{activity}):")
            print(f"  SMILES: {smiles}")
            print()

def analyze_molecular_properties(dataset, num_samples=1000):
    """Analyze molecular properties and their correlation with HIV activity"""

    print("\nAnalyzing molecular properties...")

    properties_data = []

    for i in tqdm(range(min(num_samples, len(dataset))), desc="Computing properties"):
        data = dataset[i]
        smiles = graph_to_smiles(data)

        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                properties = {
                    'hiv_active': data.y.item(),
                    'num_atoms': data.x.shape[0],
                    'num_edges': data.edge_index.shape[1] // 2,
                    'molecular_weight': Descriptors.MolWt(mol),
                    'logp': Crippen.MolLogP(mol),
                    'num_h_donors': Lipinski.NumHDonors(mol),
                    'num_h_acceptors': Lipinski.NumHAcceptors(mol),
                    'num_rotatable_bonds': Lipinski.NumRotatableBonds(mol),
                    'num_aromatic_rings': Lipinski.NumAromaticRings(mol),
                    'tpsa': Descriptors.TPSA(mol)
                }
                properties_data.append(properties)

    df = pd.DataFrame(properties_data)

    # Create property distribution plots
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()

    properties_to_plot = [
        ('num_atoms', 'Number of Atoms'),
        ('num_edges', 'Number of Edges'),
        ('molecular_weight', 'Molecular Weight (Da)'),
        ('logp', 'LogP (Lipophilicity)'),
        ('num_h_donors', 'H-Bond Donors'),
        ('num_h_acceptors', 'H-Bond Acceptors'),
        ('num_rotatable_bonds', 'Rotatable Bonds'),
        ('num_aromatic_rings', 'Aromatic Rings'),
        ('tpsa', 'Topological Polar Surface Area')
    ]

    for idx, (prop, title) in enumerate(properties_to_plot):
        ax = axes[idx]

        # Plot distributions for active and inactive
        inactive = df[df['hiv_active'] == 0][prop]
        active = df[df['hiv_active'] == 1][prop]

        # Create violin plot
        parts = ax.violinplot([inactive, active], positions=[0, 1],
                              showmeans=True, showmedians=True)

        # Customize colors
        colors = ['#ff6b6b', '#4ecdc4']
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['HIV-Inactive', 'HIV-Active'])
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(f'{title} Distribution', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add statistical test result
        from scipy import stats
        statistic, pvalue = stats.mannwhitneyu(inactive, active)
        significance = '***' if pvalue < 0.001 else '**' if pvalue < 0.01 else '*' if pvalue < 0.05 else 'ns'
        ax.text(0.5, ax.get_ylim()[1] * 0.95, f'p-value: {pvalue:.3e} ({significance})',
                ha='center', fontsize=8, fontweight='bold')

    plt.suptitle('Molecular Property Distributions: HIV-Active vs HIV-Inactive',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('molecular_properties_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print correlation analysis
    print("\n" + "="*60)
    print("PROPERTY STATISTICS (Mean ± Std)")
    print("="*60)

    for prop, title in properties_to_plot:
        inactive_mean = df[df['hiv_active'] == 0][prop].mean()
        inactive_std = df[df['hiv_active'] == 0][prop].std()
        active_mean = df[df['hiv_active'] == 1][prop].mean()
        active_std = df[df['hiv_active'] == 1][prop].std()

        print(f"{title:30s}")
        print(f"  HIV-Inactive: {inactive_mean:8.2f} ± {inactive_std:6.2f}")
        print(f"  HIV-Active:   {active_mean:8.2f} ± {active_std:6.2f}")
        print()

def visualize_graph_structure(dataset, index=0):
    """Visualize the graph structure of a molecule"""

    data = dataset[index]

    # Create NetworkX graph
    G = nx.Graph()

    # Add nodes
    for i in range(data.x.shape[0]):
        G.add_node(i)

    # Add edges
    edge_index = data.edge_index.numpy()
    for i in range(edge_index.shape[1]):
        G.add_edge(int(edge_index[0, i]), int(edge_index[1, i]))

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Graph structure
    ax1 = axes[0]
    pos = nx.spring_layout(G, k=0.5, iterations=50)

    # Color nodes by atom type (first feature)
    node_colors = [data.x[i, 0].item() for i in range(data.x.shape[0])]

    nx.draw(G, pos, ax=ax1,
            node_color=node_colors,
            cmap='tab20',
            node_size=300,
            with_labels=True,
            font_size=8,
            font_weight='bold',
            edge_color='gray',
            width=1.5)

    activity = "HIV-Active" if data.y.item() == 1 else "HIV-Inactive"
    ax1.set_title(f'Graph Structure (Molecule {index}: {activity})',
                  fontsize=14, fontweight='bold')

    # Degree distribution
    ax2 = axes[1]
    degrees = [G.degree(n) for n in G.nodes()]
    degree_counts = Counter(degrees)

    ax2.bar(degree_counts.keys(), degree_counts.values(),
            color='#3498db', edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Node Degree', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title(f'Degree Distribution', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Add statistics
    stats_text = f"Nodes: {G.number_of_nodes()}\n"
    stats_text += f"Edges: {G.number_of_edges()}\n"
    stats_text += f"Avg Degree: {np.mean(degrees):.2f}\n"
    stats_text += f"Density: {nx.density(G):.3f}"

    ax2.text(0.98, 0.98, stats_text,
             transform=ax2.transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=10)

    plt.tight_layout()
    plt.savefig(f'graph_structure_mol_{index}.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_comprehensive_report(dataset):
    """Generate a comprehensive analysis report"""

    print("\n" + "="*80)
    print(" OGB-MOLHIV DATASET COMPREHENSIVE ANALYSIS REPORT ")
    print("="*80)

    # 1. Class distribution
    visualize_class_distribution(dataset)

    # 2. Molecular structures
    visualize_molecules(dataset, max_mols=6)

    # 3. Molecular properties analysis
    analyze_molecular_properties(dataset, num_samples=1000)

    # 4. Graph structure examples
    print("\nVisualizing graph structures...")

    # Find one active and one inactive molecule
    active_idx = None
    inactive_idx = None

    for i in range(100):
        if dataset[i].y.item() == 1 and active_idx is None:
            active_idx = i
        elif dataset[i].y.item() == 0 and inactive_idx is None:
            inactive_idx = i

        if active_idx is not None and inactive_idx is not None:
            break

    if active_idx is not None:
        visualize_graph_structure(dataset, active_idx)

    if inactive_idx is not None:
        visualize_graph_structure(dataset, inactive_idx)

    print("\n" + "="*80)
    print(" ANALYSIS COMPLETE ")
    print("="*80)
    print("\nGenerated files:")
    print("  - hiv_dataset_distribution.png")
    print("  - molecular_structures.png")
    print("  - molecular_properties_analysis.png")
    print("  - graph_structure_mol_*.png")

if __name__ == "__main__":
    # Load dataset
    dataset = load_dataset_for_viz()

    # Generate comprehensive report
    create_comprehensive_report(dataset)

    # Optional: Analyze specific molecules
    print("\n" + "="*60)
    print("ANALYZING SPECIFIC MOLECULES")
    print("="*60)

    # You can specify custom indices to analyze
    custom_indices = [10, 50, 100, 200, 500, 1000]
    print(f"\nAnalyzing molecules at indices: {custom_indices}")
    visualize_molecules(dataset, indices=custom_indices, max_mols=6)
