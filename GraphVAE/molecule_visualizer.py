"""
Molecule Visualizer
Creates beautiful molecular visualizations from SMILES strings
"""

import os
import sys
import base64
from io import BytesIO
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdMolAlign, rdForceFieldHelpers
import argparse
import json


def smiles_to_image(smiles, width=400, height=400, format='PNG'):
    """
    Convert SMILES string to molecular image
    
    Args:
        smiles (str): SMILES string
        width (int): Image width
        height (int): Image height
        format (str): Image format ('PNG', 'SVG')
    
    Returns:
        str: Base64 encoded image or SVG string
    """
    try:
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Generate 2D coordinates
        Chem.rdDepictor.Compute2DCoords(mol)
        
        if format.upper() == 'SVG':
            # Create SVG drawing
            drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            svg = drawer.GetDrawingText()
            return svg
        else:
            # Create PNG image
            img = Draw.MolToImage(mol, size=(width, height))
            
            # Convert to base64
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return img_base64
            
    except Exception as e:
        print(f"Error processing SMILES '{smiles}': {e}")
        return None


def generate_3d_coordinates(smiles, num_confs=1, optimize=True, force_field='MMFF'):
    """
    Generate 3D coordinates for a molecule using ETKDG and optionally optimize them
    
    Args:
        smiles (str): SMILES string
        num_confs (int): Number of conformations to generate
        optimize (bool): Whether to optimize the structure with force field
        force_field (str): Force field to use for optimization ('MMFF' or 'UFF')
    
    Returns:
        tuple: (mol_with_3d, success_flag, error_message)
    """
    try:
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, False, "Invalid SMILES string"
        
        # Add hydrogens
        mol = Chem.AddHs(mol)
        
        # Generate 3D coordinates using ETKDG
        params = AllChem.ETKDGv3()
        params.randomSeed = 42  # For reproducibility
        params.numThreads = 0  # Use all available threads
        params.useRandomCoords = True
        params.boxSizeMult = 2.0
        params.randNegEig = True
        
        # Generate conformations
        conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
        
        if len(conf_ids) == 0:
            return None, False, "Failed to generate 3D coordinates"
        
        if optimize:
            # Optimize the structure with force field
            if force_field.upper() == 'MMFF':
                # Try MMFF94 first
                for conf_id in conf_ids:
                    try:
                        # Set up MMFF force field
                        ff = AllChem.MMFFGetMoleculeForceField(mol, 
                                                             AllChem.MMFFGetMoleculeProperties(mol), 
                                                             confId=conf_id)
                        if ff is not None:
                            ff.Minimize(maxIts=1000)
                        else:
                            # Fallback to UFF if MMFF fails
                            ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
                            if ff is not None:
                                ff.Minimize(maxIts=1000)
                    except:
                        continue
            
            elif force_field.upper() == 'UFF':
                # Use UFF force field
                for conf_id in conf_ids:
                    try:
                        ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
                        if ff is not None:
                            ff.Minimize(maxIts=1000)
                    except:
                        continue
        
        return mol, True, None
        
    except Exception as e:
        return None, False, str(e)


def mol_to_pdb_string(mol, conf_id=0):
    """
    Convert RDKit molecule to PDB string format
    
    Args:
        mol: RDKit molecule object with 3D coordinates
        conf_id (int): Conformation ID to use
    
    Returns:
        str: PDB format string
    """
    try:
        return Chem.MolToPDBBlock(mol, confId=conf_id)
    except Exception as e:
        return None


def mol_to_sdf_string(mol, conf_id=0):
    """
    Convert RDKit molecule to SDF string format
    
    Args:
        mol: RDKit molecule object with 3D coordinates
        conf_id (int): Conformation ID to use
    
    Returns:
        str: SDF format string
    """
    try:
        return Chem.MolToMolBlock(mol, confId=conf_id)
    except Exception as e:
        return None


def mol_to_xyz_string(mol, conf_id=0):
    """
    Convert RDKit molecule to XYZ string format
    
    Args:
        mol: RDKit molecule object with 3D coordinates
        conf_id (int): Conformation ID to use
    
    Returns:
        str: XYZ format string
    """
    try:
        conf = mol.GetConformer(conf_id)
        num_atoms = mol.GetNumAtoms()
        
        xyz_lines = [str(num_atoms), ""]  # Number of atoms and comment line
        
        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            symbol = atom.GetSymbol()
            xyz_lines.append(f"{symbol:<2} {pos.x:>12.6f} {pos.y:>12.6f} {pos.z:>12.6f}")
        
        return "\n".join(xyz_lines)
    except Exception as e:
        return None


def generate_3d_molecule_data(smiles, optimize=True, force_field='MMFF'):
    """
    Generate 3D molecule data in multiple formats (PDB, SDF, XYZ)
    
    Args:
        smiles (str): SMILES string
        optimize (bool): Whether to optimize the structure
        force_field (str): Force field to use for optimization
    
    Returns:
        dict: Dictionary containing 3D data in multiple formats
    """
    result = {
        'smiles': smiles,
        'success': False,
        'pdb': None,
        'sdf': None,
        'xyz': None,
        'error': None,
        'num_atoms': 0,
        'optimized': optimize,
        'force_field': force_field if optimize else None
    }
    
    try:
        # Generate 3D coordinates
        mol_3d, success, error = generate_3d_coordinates(smiles, num_confs=1, 
                                                        optimize=optimize, 
                                                        force_field=force_field)
        
        if not success:
            result['error'] = error
            return result
        
        if mol_3d is None:
            result['error'] = "Failed to generate 3D structure"
            return result
        
        # Get the number of atoms
        result['num_atoms'] = mol_3d.GetNumAtoms()
        
        # Convert to different formats
        result['pdb'] = mol_to_pdb_string(mol_3d)
        result['sdf'] = mol_to_sdf_string(mol_3d)
        result['xyz'] = mol_to_xyz_string(mol_3d)
        
        # Check if at least one format was successful
        if any([result['pdb'], result['sdf'], result['xyz']]):
            result['success'] = True
        else:
            result['error'] = "Failed to convert to any output format"
    
    except Exception as e:
        result['error'] = str(e)
    
    return result


def get_molecule_properties(smiles):
    """
    Calculate molecular properties from SMILES
    
    Args:
        smiles (str): SMILES string
    
    Returns:
        dict: Dictionary containing molecular properties
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        properties = {
            'molecular_weight': round(Descriptors.MolWt(mol), 2),
            'logp': round(Descriptors.MolLogP(mol), 2),
            'h_bond_donors': Descriptors.NumHDonors(mol),
            'h_bond_acceptors': Descriptors.NumHAcceptors(mol),
            'tpsa': round(Descriptors.TPSA(mol), 2),
            'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'aromatic_rings': Descriptors.NumAromaticRings(mol),
            'num_atoms': mol.GetNumAtoms(),
            'num_bonds': mol.GetNumBonds()
        }
        
        # Calculate drug-likeness (Lipinski's Rule of Five)
        violations = 0
        if properties['molecular_weight'] > 500:
            violations += 1
        if properties['logp'] > 5:
            violations += 1
        if properties['h_bond_donors'] > 5:
            violations += 1
        if properties['h_bond_acceptors'] > 10:
            violations += 1
        
        properties['drug_like'] = violations <= 1
        properties['lipinski_violations'] = violations
        
        return properties
        
    except Exception as e:
        print(f"Error calculating properties for SMILES '{smiles}': {e}")
        return None


def visualize_single_molecule(smiles, output_path=None, format='PNG', width=400, height=400):
    """
    Visualize a single molecule and optionally save to file
    
    Args:
        smiles (str): SMILES string
        output_path (str): Path to save image (optional)
        format (str): Image format
        width (int): Image width
        height (int): Image height
    
    Returns:
        dict: Result with image data and properties
    """
    result = {
        'smiles': smiles,
        'success': False,
        'image': None,
        'properties': None,
        'error': None
    }
    
    try:
        # Generate image
        if format.upper() == 'SVG':
            image_data = smiles_to_image(smiles, width, height, 'SVG')
        else:
            image_data = smiles_to_image(smiles, width, height, 'PNG')
        
        if image_data is None:
            result['error'] = "Could not generate image from SMILES"
            return result
        
        # Get properties
        properties = get_molecule_properties(smiles)
        
        # Save to file if requested
        if output_path:
            if format.upper() == 'SVG':
                with open(output_path, 'w') as f:
                    f.write(image_data)
            else:
                # Decode base64 and save
                img_data = base64.b64decode(image_data)
                with open(output_path, 'wb') as f:
                    f.write(img_data)
        
        result.update({
            'success': True,
            'image': image_data,
            'properties': properties
        })
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def batch_visualize_smiles(smiles_file, output_dir=None, format='PNG', width=400, height=400):
    """
    Visualize multiple molecules from a SMILES file
    
    Args:
        smiles_file (str): Path to file containing SMILES (one per line)
        output_dir (str): Directory to save images (optional)
        format (str): Image format
        width (int): Image width
        height (int): Image height
    
    Returns:
        list: List of results for each molecule
    """
    results = []
    
    try:
        with open(smiles_file, 'r') as f:
            smiles_list = [line.strip() for line in f if line.strip()]
        
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for i, smiles in enumerate(smiles_list):
            print(f"Processing molecule {i+1}/{len(smiles_list)}: {smiles}")
            
            output_path = None
            if output_dir:
                ext = 'svg' if format.upper() == 'SVG' else 'png'
                filename = f"molecule_{i+1:03d}.{ext}"
                output_path = os.path.join(output_dir, filename)
            
            result = visualize_single_molecule(smiles, output_path, format, width, height)
            result['index'] = i
            results.append(result)
    
    except Exception as e:
        print(f"Error reading SMILES file: {e}")
        return []
    
    return results


def create_molecule_grid(smiles_list, output_path, molecules_per_row=4, mol_size=(200, 200)):
    """
    Create a grid of molecules from SMILES list
    
    Args:
        smiles_list (list): List of SMILES strings
        output_path (str): Path to save grid image
        molecules_per_row (int): Number of molecules per row
        mol_size (tuple): Size of each molecule image
    
    Returns:
        bool: Success status
    """
    try:
        mols = []
        labels = []
        
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mols.append(mol)
                # Truncate long SMILES for labels
                label = smiles[:20] + '...' if len(smiles) > 20 else smiles
                labels.append(f"{i+1}: {label}")
        
        if not mols:
            print("No valid molecules found")
            return False
        
        # Create grid image
        img = Draw.MolsToGridImage(
            mols, 
            molsPerRow=molecules_per_row,
            subImgSize=mol_size,
            legends=labels
        )
        
        img.save(output_path)
        print(f"Grid saved to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error creating molecule grid: {e}")
        return False


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Molecule Visualizer from SMILES')
    parser.add_argument('--smiles', type=str, help='Single SMILES string to visualize')
    parser.add_argument('--file', type=str, help='File containing SMILES (one per line)')
    parser.add_argument('--output', type=str, help='Output file/directory path')
    parser.add_argument('--format', type=str, default='PNG', choices=['PNG', 'SVG'], 
                        help='Output format')
    parser.add_argument('--width', type=int, default=400, help='Image width')
    parser.add_argument('--height', type=int, default=400, help='Image height')
    parser.add_argument('--grid', action='store_true', help='Create molecule grid for batch processing')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    
    args = parser.parse_args()
    
    if args.smiles:
        # Single molecule
        result = visualize_single_molecule(
            args.smiles, 
            args.output, 
            args.format, 
            args.width, 
            args.height
        )
        
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            if result['success']:
                print(f"Successfully visualized: {args.smiles}")
                if result['properties']:
                    props = result['properties']
                    print(f"Molecular Weight: {props['molecular_weight']}")
                    print(f"LogP: {props['logp']}")
                    print(f"Drug-like: {props['drug_like']}")
            else:
                print(f"Failed to visualize: {result['error']}")
    
    elif args.file:
        # Batch processing
        if args.grid and args.output:
            # Read SMILES and create grid
            try:
                with open(args.file, 'r') as f:
                    smiles_list = [line.strip() for line in f if line.strip()]
                
                success = create_molecule_grid(smiles_list, args.output)
                if success:
                    print(f"Grid created successfully: {args.output}")
                else:
                    print("Failed to create grid")
            except Exception as e:
                print(f"Error: {e}")
        else:
            # Individual images
            results = batch_visualize_smiles(
                args.file,
                args.output,
                args.format,
                args.width,
                args.height
            )
            
            if args.json:
                print(json.dumps(results, indent=2))
            else:
                successful = sum(1 for r in results if r['success'])
                print(f"Successfully processed {successful}/{len(results)} molecules")
    
    else:
        # Interactive mode
        print("Molecule Visualizer - Interactive Mode")
        print("Enter SMILES strings (or 'quit' to exit):")
        
        while True:
            smiles = input("SMILES: ").strip()
            
            if smiles.lower() in ['quit', 'exit', 'q']:
                break
            
            if not smiles:
                continue
            
            result = visualize_single_molecule(smiles)
            
            if result['success']:
                print(f"✓ Valid molecule!")
                if result['properties']:
                    props = result['properties']
                    print(f"  MW: {props['molecular_weight']}, LogP: {props['logp']}")
                    print(f"  Drug-like: {props['drug_like']}")
            else:
                print(f"✗ Error: {result['error']}")


if __name__ == "__main__":
    main()
