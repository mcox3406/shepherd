"""
Basic tests for the inference scaling components of ShEPhERD.
"""

import sys
import os
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from shepherd.inference_scaling import (
    ShepherdModelRunner,
    Verifier,
    SAScoreVerifier,
    CLogPVerifier,
    QEDVerifier,
    MultiObjectiveVerifier,
    RandomSearch,
    ZeroOrderSearch,
    GuidedSearch,
    create_rdkit_molecule
)


class MockMolecule:
    """Mock RDKit molecule for testing verifiers without RDKit."""
    
    def __init__(self, sa_score=3.0, logp=2.5, qed=0.75):
        self.sa_score = sa_score
        self.logp = logp
        self.qed = qed


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    return MagicMock()


@pytest.fixture
def mock_inference_sample():
    """Create a mock inference_sample function."""
    mock = MagicMock()
    mock.return_value = [{"x1": {"atoms": np.array([6, 6, 8]), "positions": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])}}]
    return mock


@pytest.fixture
def mock_sample():
    """Create a mock sample for testing verifiers."""
    return {
        "x1": {
            "atoms": np.array([6, 6, 8]),  # C, C, O
            "positions": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        }
    }


@pytest.fixture
def mock_mol():
    """Create a mock RDKit molecule."""
    return MockMolecule(sa_score=3.0, logp=2.5)


@pytest.fixture
def mock_runner():
    """Create a mock model runner."""
    mock = MagicMock()
    mock.return_value = {"x1": {"atoms": np.array([6, 6, 8]), "positions": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])}}
    mock.get_last_noise = MagicMock(return_value=torch.randn(10))
    return mock


@pytest.fixture
def mock_verifier():
    """Create a mock verifier."""
    mock = MagicMock()
    mock.return_value = 0.75
    return mock


# tests for ShepherdModelRunner
@patch('shepherd.inference.inference_sample')
def test_model_runner_without_custom_noise(mock_inference, mock_model, mock_inference_sample):
    """Test that the model runner works without custom noise."""
    mock_inference.return_value = mock_inference_sample.return_value
    
    # create model runner
    runner = ShepherdModelRunner(
        model_pl=mock_model,
        batch_size=1,
        N_x1=3,
        N_x4=1,
        unconditional=True
    )
    
    result = runner()
    
    mock_inference.assert_called_once()
    
    assert "x1" in result
    assert len(result["x1"]["atoms"]) == 3


@patch('shepherd.inference.inference_sample')
def test_model_runner_with_custom_noise(mock_inference, mock_model, mock_inference_sample):
    """Test that the model runner works with custom noise."""
    mock_inference.return_value = mock_inference_sample.return_value
    
    # create model runner
    runner = ShepherdModelRunner(
        model_pl=mock_model,
        batch_size=1,
        N_x1=3,
        N_x4=1,
        unconditional=True
    )
    
    custom_noise = torch.randn(10)
    
    result = runner(noise=custom_noise)
    
    mock_inference.assert_called_once()
    
    assert runner.get_last_noise() is not None
    
    assert "x1" in result
    assert len(result["x1"]["atoms"]) == 3


# tests for verifiers
@patch('shepherd.inference_scaling.verifiers.create_rdkit_molecule')
def test_verifier_preprocess(mock_create_molecule, mock_sample, mock_mol):
    """Test that the verifier base class can preprocess ShEPhERD output."""
    mock_create_molecule.return_value = mock_mol
    
    verifier = Verifier("test")
    
    mol = verifier.preprocess(mock_sample)
    
    mock_create_molecule.assert_called_once_with(mock_sample)
    assert mol is mock_mol


@patch.object(SAScoreVerifier, '_load_sa_model')
@patch.object(SAScoreVerifier, 'preprocess')
def test_sa_score_verifier(mock_preprocess, mock_load_sa_model, mock_sample, mock_mol):
    """Test the SA score verifier."""
    # mock the preprocess method to return our mock molecule
    mock_preprocess.return_value = mock_mol
    
    # set up the SA_Score mock
    mock_sascorer = MagicMock()
    mock_sascorer.calculateScore.return_value = 3.0
    
    verifier = SAScoreVerifier()
    verifier._sascorer = mock_sascorer
    
    score = verifier(mock_sample)
    
    mock_preprocess.assert_called_once_with(mock_sample)
    mock_sascorer.calculateScore.assert_called_once_with(mock_mol)
    
    # check that the score is as expected (normalized SA score)
    # SA score of 3.0 should be normalized to ~0.78
    assert score > 0.7
    assert score < 0.8


@patch.object(CLogPVerifier, 'preprocess')
def test_clogp_verifier(mock_preprocess, mock_sample, mock_mol):
    """Test the cLogP verifier with continuous scaling."""
    # mock the preprocess method to return our mock molecule
    mock_preprocess.return_value = mock_mol
    
    # test with a value in the middle of the range
    with patch('rdkit.Chem.Crippen.MolLogP', return_value=1.5):
        verifier = CLogPVerifier()  # default range is (-1, 4)
        
        score = verifier(mock_sample)
        
        mock_preprocess.assert_called_once_with(mock_sample)
        
        # check that the score is as expected
        # For logP=1.5 with range (-1, 4), expected score is (1.5 - (-1))/(4-(-1)) = 2.5/5 = 0.5
        assert score == 0.5
    
    # reset the mock
    mock_preprocess.reset_mock()
    
    # test with a value at lower bound
    with patch('rdkit.Chem.Crippen.MolLogP', return_value=-1.0):
        verifier = CLogPVerifier()
        score = verifier(mock_sample)
        assert score == 0.0  # lower bound should map to 0.0
    
    # reset the mock
    mock_preprocess.reset_mock()
    
    # test with a value at upper bound
    with patch('rdkit.Chem.Crippen.MolLogP', return_value=4.0):
        verifier = CLogPVerifier()
        score = verifier(mock_sample)
        assert score == 1.0  # upper bound should map to 1.0
    
    # reset the mock
    mock_preprocess.reset_mock()
    
    # test with a value outside the range (lower)
    with patch('rdkit.Chem.Crippen.MolLogP', return_value=-2.0):
        verifier = CLogPVerifier()
        score = verifier(mock_sample)
        assert score == 0.0  # below lower bound should be clamped to 0.0
    
    # reset the mock
    mock_preprocess.reset_mock()
    
    # test with a value outside the range (upper)
    with patch('rdkit.Chem.Crippen.MolLogP', return_value=5.0):
        verifier = CLogPVerifier()
        score = verifier(mock_sample)
        assert score == 1.0  # above upper bound should be clamped to 1.0


@patch.object(QEDVerifier, 'preprocess')
def test_qed_verifier(mock_preprocess, mock_sample, mock_mol):
    """Test the QED verifier."""
    # mock the preprocess method to return our mock molecule
    mock_preprocess.return_value = mock_mol
    
    # mock RDKit's QED.qed
    with patch('rdkit.Chem.QED.qed', return_value=0.75):
        verifier = QEDVerifier()
        
        score = verifier(mock_sample)
        
        mock_preprocess.assert_called_once_with(mock_sample)
        
        # check that the score is as expected (QED is already 0-1)
        assert score == 0.75
        
        # test with custom weights
        custom_weights = (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
        verifier_custom = QEDVerifier(custom_weights=custom_weights)
        
        with patch('rdkit.Chem.QED.qed', return_value=0.6) as mock_qed:
            score_custom = verifier_custom(mock_sample)
            mock_qed.assert_called_once_with(mock_mol, w=custom_weights)
            assert score_custom == 0.6


def test_multi_objective_verifier(mock_sample):
    """Test the multi-objective verifier."""
    # create mock verifiers
    mock_verifier1 = MagicMock()
    mock_verifier1.return_value = 0.8
    mock_verifier1.weight = 1.0
    
    mock_verifier2 = MagicMock()
    mock_verifier2.return_value = 0.6
    mock_verifier2.weight = 1.0
    
    # create the multi-objective verifier
    verifier = MultiObjectiveVerifier([mock_verifier1, mock_verifier2])
    
    # calculate the score
    score = verifier(mock_sample)
    
    # check that both verifiers were called
    mock_verifier1.assert_called_once_with(mock_sample)
    mock_verifier2.assert_called_once_with(mock_sample)
    
    # check that the score is as expected (weighted average)
    assert score == 0.7


# tests for search algorithms
def test_random_search(mock_runner, mock_verifier):
    """Test the random search algorithm."""
    # create a search algorithm
    search = RandomSearch(mock_verifier, mock_runner)
    
    # run the search with a small number of trials
    best_sample, best_score, scores, metadata = search.search(num_trials=3, verbose=False)
    
    assert mock_runner.call_count == 3
    
    assert mock_verifier.call_count == 3
    
    assert best_score == 0.75
    
    assert len(scores) == 3


def test_zero_order_search(mock_runner, mock_verifier):
    """Test the zero-order search algorithm."""
    # create a search algorithm
    search = ZeroOrderSearch(mock_verifier, mock_runner)
    
    # run the search with a small number of steps
    best_sample, best_score, scores, metadata = search.search(
        num_steps=2, 
        num_neighbors=2, 
        verbose=False
    )
    
    # check that the model runner was called the expected number of times
    # initial sample + (num_steps * num_neighbors)
    expected_calls = 1 + (2 * 2)
    assert mock_runner.call_count == expected_calls
    
    # check that the verifier was called the expected number of times
    assert mock_verifier.call_count == expected_calls
    
    # check that the best score is as expected
    assert best_score == 0.75
    
    # check that the scores history has the expected length
    # initial score + num_steps
    assert len(scores) == 3


def test_guided_search(mock_runner, mock_verifier):
    """Test the guided search algorithm."""
    # create a search algorithm
    search = GuidedSearch(mock_verifier, mock_runner)
    
    # run the search with a small population and few generations
    best_sample, best_score, scores, metadata = search.search(
        pop_size=3, 
        num_generations=2, 
        verbose=False
    )
    
    # check that the model runner was called the expected number of times
    # just check that it's at least the size of the population
    assert mock_runner.call_count >= 3
    
    assert mock_verifier.call_count >= mock_runner.call_count
    
    assert best_score == 0.75
    
    # check that the scores history has the expected length
    # initial score + num_generations
    assert len(scores) == 3 