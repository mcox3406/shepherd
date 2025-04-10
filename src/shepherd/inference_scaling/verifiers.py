"""
Verifiers for inference-time scaling with ShEPhERD.

This module implements verifiers that can be used for inference-time scaling
of the ShEPhERD model. Verifiers are functions that evaluate the quality of 
generated molecules according to specific criteria (e.g., SA score, solubility, 
cLogP).
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import RDConfig
import os
import sys
import logging

# turn off RDKit warnings or else you'll be spammed
# uncomment this if you're testing stuff though
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from .utils import create_rdkit_molecule


class Verifier:
    """Base class for all verifiers."""
    
    def __init__(self, name, weight=1.0):
        """
        Initialize a verifier.
        
        Args:
            name (str): Name of the verifier.
            weight (float): Weight of the verifier in multi-objective optimization.
        """
        self.name = name
        self.weight = weight
        
    def __call__(self, mol):
        """
        Evaluate the molecule.
        
        Args:
            mol: The molecule to evaluate. This can be an RDKit mol object or a
                 dictionary containing the ShEPhERD generation output.
                 
        Returns:
            float: The score of the molecule. Higher scores indicate better molecules.
        """
        raise NotImplementedError("Subclasses must implement the __call__ method.")
    
    def preprocess(self, shepherd_output):
        """
        Preprocess ShEPhERD output to extract RDKit mol object.
        
        Args:
            shepherd_output (dict): Dictionary output from ShEPhERD generation.
            
        Returns:
            rdkit.Chem.rdchem.Mol: RDKit molecule object.
        """
        return create_rdkit_molecule(shepherd_output)


class SAScoreVerifier(Verifier):
    """Synthetic accessibility score verifier."""
    
    def __init__(self, weight=1.0, target_range=(1.0, 3.5)):
        """
        Initialize the SA Score verifier.
        
        Args:
            weight (float): Weight of the verifier in multi-objective optimization.
            target_range (tuple): The ideal range for SA scores. Scores outside
                                 this range will be penalized.
        """
        super().__init__("SA_Score", weight)
        self.target_range = target_range
        self._sascorer = None
        
    def _load_sa_model(self):
        """Load the SA score model if it hasn't been loaded yet."""
        if self._sascorer is None:
            # import sascorer from RDKit contrib directory
            # https://greglandrum.github.io/rdkit-blog/posts/2023-12-01-using_sascore_and_npscore.html
            sys.path.append(os.path.join(os.environ['CONDA_PREFIX'],'share','RDKit','Contrib'))
            try:
                from SA_Score import sascorer
                self._sascorer = sascorer
            except ImportError:
                try:
                    from rdkit.Contrib.SA_Score import sascorer
                    self._sascorer = sascorer
                except ImportError:
                    logging.error("Failed to import sascorer from RDKit contrib directory")
                    raise
            
    def __call__(self, mol_input):
        """
        Calculate the SA score for a molecule.
        
        Args:
            mol_input: RDKit Mol object or ShEPhERD generation output.
            
        Returns:
            float: Normalized SA score between 0 and 1, where higher is better.
        """
        if isinstance(mol_input, dict):
            mol = self.preprocess(mol_input)
        else:
            mol = mol_input
            
        if mol is None:
            return 0.0
        
        try:
            self._load_sa_model()
            
            sa_score = self._sascorer.calculateScore(mol)
            
            # original SA score is 1 (easy to synthesize) to 10 (hard to synthesize)
            # we want to normalize it to 0-1 where higher is better (more synthesizable)
            
            # invert and normalize to 0-1
            normalized_score = 1.0 - (sa_score - 1.0) / 9.0  # 1->1, 10->0
            
            # WE MIGHT WANT TO USE THIS TOO (UNCOMMENT IF YOU WANT TO USE IT)
            # min_range, max_range = self.target_range
            # original_range_score = 1.0 - (sa_score - min_range) / (max_range - min_range)
            # if sa_score < min_range:
            #     # small penalty for being too easy to synthesize (might be too simple)
            #     normalized_score *= 0.9
            # elif sa_score > max_range:
            #     # larger penalty for being too hard to synthesize
            #     normalized_score *= 0.7
            
            return normalized_score
        
        except Exception as e:
            logging.warning(f"Error calculating SA score: {e}")
            return 0.0


class CLogPVerifier(Verifier):
    """Verifier for calculated LogP values in a druglike range."""
    
    def __init__(self, weight=1.0, target_range=(0.4, 5.0)):
        """
        Initialize the cLogP verifier.
        
        Args:
            weight (float): Weight of the verifier in multi-objective optimization.
            target_range (tuple): The ideal range for cLogP values. Values outside
                                  this range will be penalized.
        """
        super().__init__("cLogP", weight)
        self.target_range = target_range
        from rdkit.Chem import Crippen
        self._crippen = Crippen
        
    def __call__(self, mol_input):
        """
        Calculate a score based on how well the cLogP value fits in a druglike range.
        
        Args:
            mol_input: RDKit Mol object or ShEPhERD generation output.
            
        Returns:
            float: Score between 0 and 1, where higher is better.
        """
        if isinstance(mol_input, dict):
            mol = self.preprocess(mol_input)
        else:
            mol = mol_input
            
        if mol is None:
            return 0.0
        
        try:
            logp = self._crippen.MolLogP(mol)
            
            min_range, max_range = self.target_range
            if min_range <= logp <= max_range:
                # within range gets full score
                return 1.0
            elif logp < min_range:
                # too hydrophilic
                distance = min_range - logp
                penalty = min(1.0, distance / 2.0)  # gradually reduce score
                return 1.0 - penalty
            else:  # logp > max_range
                # too lipophilic
                distance = logp - max_range
                penalty = min(1.0, distance / 3.0)  # gradually reduce score
                return 1.0 - penalty
        
        except Exception as e:
            logging.warning(f"Error calculating cLogP: {e}")
            return 0.0


class MultiObjectiveVerifier(Verifier):
    """Combines multiple verifiers for multi-objective optimization."""
    
    def __init__(self, verifiers, aggregation="weighted_sum"):
        """
        Initialize a multi-objective verifier.
        
        Args:
            verifiers (list): List of Verifier objects.
            aggregation (str): Aggregation method. One of "weighted_sum", "product", "min".
        """
        super().__init__("MultiObjective")
        self.verifiers = verifiers
        self.aggregation = aggregation
        
    def __call__(self, mol_input):
        """
        Evaluate the molecule using all verifiers.
        
        Args:
            mol_input: The molecule to evaluate.
            
        Returns:
            float: The aggregated score.
        """
        scores = []
        weights = []
        
        # get scores from all verifiers
        for verifier in self.verifiers:
            score = verifier(mol_input)
            scores.append(score)
            weights.append(verifier.weight)
            
        # normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # aggregate scores
        if self.aggregation == "weighted_sum":
            return np.sum(np.array(scores) * weights)
        elif self.aggregation == "product":
            return np.prod(np.array(scores) ** weights)
        elif self.aggregation == "min":
            return np.min(scores)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}") 