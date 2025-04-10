"""
Simple script to analyze the content of a pickle file containing generated molecules from ShEPhERD.
"""

import pickle
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy

from rdkit import Chem
from rdkit.Chem import Draw, AllChem, rdDetermineBonds


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze pickle file with ShEPhERD molecule samples')
    parser.add_argument('--pickle_file', type=str, default='basic_inference_results/samples.pickle',
                        help='Path to the pickle file with samples')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save output files (defaults to same directory as pickle file)')
    return parser.parse_args()

def examine_atom_data(sample):
    if 'x1' not in sample:
        print("No atom data (x1) found in sample")
        return
    
    print("ATOM DATA (x1):")
    
    atoms = sample['x1']['atoms']
    positions = sample['x1']['positions']
    
    print(f"Number of atoms: {len(atoms)}")
    print(f"Atom types: {atoms}")
    

def create_rdkit_molecule(sample):
    if 'x1' not in sample:
        print("No atom data (x1) found in sample")
        return None
    
    print("RDKIT MOLECULE CREATION:")
    
    atoms = sample['x1']['atoms']
    positions = sample['x1']['positions']
    
    try:
        # create XYZ format string
        xyz = f'{len(atoms)}\n\n'
        for a in range(len(atoms)):
            atomic_number = int(atoms[a])
            position = positions[a]
            symbol = Chem.Atom(atomic_number).GetSymbol()
            xyz += f'{symbol} {position[0]:.3f} {position[1]:.3f} {position[2]:.3f}\n'
        
        # create molecule from XYZ block
        mol = Chem.MolFromXYZBlock(xyz)
        if mol is None:
            print("Failed to create molecule!")
            return None
        
        # try different charge states for bond determination
        mol_final = None
        for charge in [0, 1, -1, 2, -2]:
            mol_copy = deepcopy(mol)
            try:
                rdDetermineBonds.DetermineBonds(mol_copy, charge=charge, embedChiral=True)
                print(f"Bond determination successful with charge {charge}")
                mol_final = mol_copy
                break
            except Exception as e:
                # print(f"Bond determination failed with charge {charge}: {e}")
                continue
        
        if mol_final is None:
            print("Bond determination failed for all charge states")
            return None
        
        # validate molecule
        try:
            radical_electrons = sum([a.GetNumRadicalElectrons() for a in mol_final.GetAtoms()])
            if radical_electrons > 0:
                print(f"Warning: Molecule has {radical_electrons} radical electrons")
            
            mol_final.UpdatePropertyCache()
            Chem.GetSymmSSSR(mol_final)
            print("Molecule validation successful")
        except Exception as e:
            print(f"Molecule validation failed: {e}")
            return None
        
        # try to generate SMILES to verify molecule
        try:
            smiles = Chem.MolToSmiles(mol_final)
            print(f"Generated SMILES: {smiles}")
        except Exception as e:
            print(f"SMILES generation failed: {e}")
        
        return mol_final
        
    except Exception as e:
        print(f"Error creating molecule: {e}")
        return None

def main():
    args = parse_args()
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.pickle_file).parent / 'analysis'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading samples from {args.pickle_file}")
    try:
        with open(args.pickle_file, 'rb') as f:
            samples = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return
    
    if isinstance(samples, list):
        print(f"Loaded {len(samples)} samples")
    else:
        print("Loaded a single sample")
        samples = [samples]
    
    for i, sample in enumerate(samples):
        print(f"\n\n==========================================")
        print(f"SAMPLE {i+1}")
        print(f"==========================================")
        
        examine_atom_data(sample)
        
        mol = create_rdkit_molecule(sample)
        if mol:
            try:
                smi = Chem.MolToSmiles(mol)
                with open(output_dir / f"molecule_{i+1}.smi", "w") as f:
                    f.write(smi)
            except Exception as e:
                print(f"Error generating SMILES: {e}")
            
            # save 2D image
            try:
                img = Draw.MolToImage(mol, size=(400, 400))
                img_path = output_dir / f"molecule_{i+1}_2d.png"
                img.save(str(img_path))
                print(f"Saved 2D molecule image to {img_path}")
            except Exception as e:
                print(f"Error creating 2D image: {e}")
            
            # save 3D model (PDB)
            try:
                pdb_path = str(output_dir / f"molecule_{i+1}.pdb")
                Chem.MolToPDBFile(mol, pdb_path)
                print(f"Saved 3D structure to {pdb_path}")
            except Exception as e:
                print(f"Error creating 3D model: {e}")

if __name__ == "__main__":
    main()