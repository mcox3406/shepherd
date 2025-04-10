"""
Utility functions for ShEPhERD inference scaling.

This module provides helper functions for working with ShEPhERD's inference output.
"""

import logging
from copy import deepcopy

try:
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
    rdkit_available = True
except ImportError:
    rdkit_available = False
    logging.warning("RDKit not available, molecule processing functions will not work")


def create_rdkit_molecule(sample):
    """
    Create an RDKit molecule from ShEPhERD output using XYZ block approach.
    
    Args:
        sample (dict): ShEPhERD output dictionary with x1 containing atoms and positions.
        
    Returns:
        rdkit.Chem.rdchem.Mol: RDKit molecule object or None if conversion fails.
    """
    if not rdkit_available:
        logging.error("RDKit not available, cannot create molecule")
        return None
        
    if 'x1' not in sample:
        logging.warning("No atom data (x1) found in sample")
        return None
    
    try:
        # extract atoms and their positions from x1
        atoms = sample['x1']['atoms']
        positions = sample['x1']['positions']
        
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
            logging.warning("Failed to create molecule from XYZ block")
            return None
        
        # try different charge states for bond determination
        mol_final = None
        for charge in [0, 1, -1, 2, -2]:
            mol_copy = deepcopy(mol)
            try:
                rdDetermineBonds.DetermineBonds(mol_copy, charge=charge, embedChiral=True)
                logging.debug(f"Bond determination successful with charge {charge}")
                mol_final = mol_copy
                break
            except Exception as e:
                logging.debug(f"Bond determination failed with charge {charge}: {e}")
                continue
        
        if mol_final is None:
            logging.warning("Bond determination failed for all charge states")
            return None
        
        # validate molecule
        try:
            radical_electrons = sum([a.GetNumRadicalElectrons() for a in mol_final.GetAtoms()])
            if radical_electrons > 0:
                logging.warning(f"Molecule has {radical_electrons} radical electrons")
            
            mol_final.UpdatePropertyCache()
            Chem.GetSymmSSSR(mol_final)
            logging.debug("Molecule validation successful")
        except Exception as e:
            logging.warning(f"Molecule validation failed: {e}")
            return None
        
        # try to generate SMILES to verify molecule
        try:
            smiles = Chem.MolToSmiles(mol_final)
            logging.debug(f"Generated SMILES: {smiles}")
        except Exception as e:
            logging.warning(f"SMILES generation failed: {e}")
        
        return mol_final
        
    except Exception as e:
        logging.warning(f"Error creating molecule: {e}")
        return None 