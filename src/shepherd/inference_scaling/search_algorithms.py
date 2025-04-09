"""
Search algorithms for inference-time scaling with ShEPhERD.
"""

import numpy as np
import torch
import logging
import time
from tqdm import tqdm


class SearchAlgorithm:
    """Base class for all search algorithms."""
    
    def __init__(self, verifier, model_runner, name=None):
        """
        Initialize a search algorithm.
        
        Args:
            verifier: A verifier callable that evaluates molecular quality.
            model_runner: A callable that runs the diffusion model with a given noise.
            name (str, optional): Name of the search algorithm.
        """
        self.verifier = verifier
        self.model_runner = model_runner
        self.name = name or self.__class__.__name__
        
    def search(self, *args, **kwargs):
        """
        Run the search algorithm.
        
        Returns:
            tuple: (best_sample, best_score, scores_history, metadata)
        """
        raise NotImplementedError("Subclasses must implement the search method.")


class RandomSearch(SearchAlgorithm):
    """
    Random search algorithm for inference-time scaling.
    
    Generates N random noise vectors, evaluates them, and returns the best one.
    This is the simplest baseline approach described in Ma et al. (2025).
    """
    
    def __init__(self, verifier, model_runner, name="RandomSearch"):
        """
        Initialize a random search algorithm.
        
        Args:
            verifier: A verifier callable that evaluates molecular quality.
            model_runner: A callable that runs the diffusion model with a given noise.
            name (str, optional): Name of the search algorithm.
        """
        super().__init__(verifier, model_runner, name)
        
    def search(self, num_trials=50, verbose=True, device="cpu"):
        """
        Run random search to find the best noise vector.
        
        Args:
            num_trials (int): Number of random noise vectors to try.
            verbose (bool): Whether to print progress information.
            device (str): Device to run the model on ("cpu" or "cuda").
            
        Returns:
            tuple: (best_sample, best_score, scores_history, metadata)
        """
        best_score = -float('inf')
        best_sample = None
        scores = []
        times = []
        
        if verbose:
            iterator = tqdm(range(num_trials), desc="Random Search")
        else:
            iterator = range(num_trials)
        
        for i in iterator:
            start_time = time.time()
            
            # generate sample with a random noise vector
            sample = self.model_runner()
            
            # evaluate sample
            score = self.verifier(sample)
            scores.append(score)
            
            # update best
            if score > best_score:
                best_score = score
                best_sample = sample
                if verbose:
                    logging.info(f"Trial {i+1}/{num_trials}: New best score: {best_score:.4f}")
            
            times.append(time.time() - start_time)
        
        metadata = {
            "mean_time_per_trial": np.mean(times),
            "total_time": sum(times),
            "num_trials": num_trials
        }
        
        return best_sample, best_score, scores, metadata


class ZeroOrderSearch(SearchAlgorithm):
    """
    Zero-order search algorithm for inference-time scaling.
    
    Implements a simple zero-order optimization method that perturbs a pivot noise
    vector and moves in the direction that improves the score.
    """
    
    def __init__(self, verifier, model_runner, name="ZeroOrderSearch"):
        """
        Initialize a zero-order search algorithm.
        
        Args:
            verifier: A verifier callable that evaluates molecular quality.
            model_runner: A callable that runs the diffusion model with a given noise.
            name (str, optional): Name of the search algorithm.
        """
        super().__init__(verifier, model_runner, name)
        
    def search(self, num_steps=20, num_neighbors=5, step_size=0.1, 
               init_noise=None, verbose=True, device="cpu"):
        """
        Run zero-order search to find the best noise vector.
        
        Args:
            num_steps (int): Number of optimization steps.
            num_neighbors (int): Number of neighbors to evaluate at each step.
            step_size (float): Size of perturbation for generating neighbors.
            init_noise (torch.Tensor, optional): Initial noise vector. If None,
                                                a random noise vector is used.
            verbose (bool): Whether to print progress information.
            device (str): Device to run the model on ("cpu" or "cuda").
            
        Returns:
            tuple: (best_sample, best_score, scores_history, metadata)
        """
        if init_noise is None:
            # get a random noise vector from the model_runner
            init_sample = self.model_runner()
            init_score = self.verifier(init_sample)
            pivot_noise = self.model_runner.get_last_noise()
            pivot_sample = init_sample
            pivot_score = init_score
        else:
            pivot_noise = init_noise
            pivot_sample = self.model_runner(noise=pivot_noise)
            pivot_score = self.verifier(pivot_sample)
        
        scores = [pivot_score]
        times = []
        
        if verbose:
            iterator = tqdm(range(num_steps), desc="Zero-Order Search")
        else:
            iterator = range(num_steps)
        
        for step in iterator:
            start_time = time.time()
            
            # generate neighbors
            neighbor_samples = []
            neighbor_scores = []
            
            for n in range(num_neighbors):
                # add random perturbation to the pivot noise
                perturbation = torch.randn_like(pivot_noise) * step_size
                neighbor_noise = pivot_noise + perturbation
                
                # generate sample with the perturbed noise
                neighbor_sample = self.model_runner(noise=neighbor_noise)
                
                # evaluate sample
                neighbor_score = self.verifier(neighbor_sample)
                
                neighbor_samples.append(neighbor_sample)
                neighbor_scores.append(neighbor_score)
            
            # find the best neighbor
            best_idx = np.argmax(neighbor_scores)
            best_neighbor_score = neighbor_scores[best_idx]
            
            # update pivot if the best neighbor is better
            if best_neighbor_score > pivot_score:
                pivot_sample = neighbor_samples[best_idx]
                pivot_score = best_neighbor_score
                pivot_noise = self.model_runner.get_last_noise()
                
                if verbose:
                    logging.info(f"Step {step+1}/{num_steps}: New best score: {pivot_score:.4f}")
            
            scores.append(pivot_score)
            times.append(time.time() - start_time)
        
        metadata = {
            "mean_time_per_step": np.mean(times),
            "total_time": sum(times),
            "num_steps": num_steps,
            "num_neighbors": num_neighbors,
            "step_size": step_size
        }
        
        return pivot_sample, pivot_score, scores, metadata


class GuidedSearch(SearchAlgorithm):
    """
    Guided search algorithm for inference-time scaling.
    
    This method starts with multiple independent random noise vectors and gradually
    evolves them towards better solutions.
    """
    
    def __init__(self, verifier, model_runner, name="GuidedSearch"):
        """
        Initialize a guided search algorithm.
        
        Args:
            verifier: A verifier callable that evaluates molecular quality.
            model_runner: A callable that runs the diffusion model with a given noise.
            name (str, optional): Name of the search algorithm.
        """
        super().__init__(verifier, model_runner, name)
        
    def search(self, pop_size=10, num_generations=5, mutation_rate=0.2, 
               elite_fraction=0.2, verbose=True, device="cpu"):
        """
        Run guided search to find the best noise vector.
        
        Args:
            pop_size (int): Number of noise vectors in the population.
            num_generations (int): Number of generations to evolve.
            mutation_rate (float): Rate of mutation for generating new candidates.
            elite_fraction (float): Fraction of the population to keep as elite.
            verbose (bool): Whether to print progress information.
            device (str): Device to run the model on ("cpu" or "cuda").
            
        Returns:
            tuple: (best_sample, best_score, scores_history, metadata)
        """
        # initialize population
        population = []
        scores = []
        
        if verbose:
            print("Initializing population...")
        
        for i in range(pop_size):
            sample = self.model_runner()
            score = self.verifier(sample)
            noise = self.model_runner.get_last_noise()
            
            population.append((sample, noise, score))
            scores.append(score)
        
        # sort by score (descending)
        population.sort(key=lambda x: x[2], reverse=True)
        best_sample, best_noise, best_score = population[0]
        
        history = [best_score]
        times = []
        
        elite_count = max(1, int(pop_size * elite_fraction))
        
        if verbose:
            iterator = tqdm(range(num_generations), desc="Guided Search")
        else:
            iterator = range(num_generations)
        
        for gen in iterator:
            start_time = time.time()
            
            # select elite individuals
            elite = population[:elite_count]
            
            # generate new population
            new_population = []
            
            # keep elite individuals
            new_population.extend(elite)
            
            # generate remaining individuals
            while len(new_population) < pop_size:
                # select a parent (weighted by rank)
                parent_idx = np.random.choice(
                    len(population), 
                    p=np.array([1/(i+1) for i in range(len(population))]) / sum(1/(i+1) for i in range(len(population)))
                )
                _, parent_noise, _ = population[parent_idx]
                
                # apply mutation
                mutation = torch.randn_like(parent_noise) * mutation_rate
                new_noise = parent_noise + mutation
                
                # generate sample with the new noise
                new_sample = self.model_runner(noise=new_noise)
                new_score = self.verifier(new_sample)
                new_noise = self.model_runner.get_last_noise()
                
                new_population.append((new_sample, new_noise, new_score))
            
            # update population
            population = new_population
            
            # sort by score (descending)
            population.sort(key=lambda x: x[2], reverse=True)
            
            # update best
            if population[0][2] > best_score:
                best_sample, best_noise, best_score = population[0]
                if verbose:
                    logging.info(f"Generation {gen+1}/{num_generations}: New best score: {best_score:.4f}")
            
            history.append(best_score)
            times.append(time.time() - start_time)
        
        metadata = {
            "mean_time_per_generation": np.mean(times),
            "total_time": sum(times),
            "num_generations": num_generations,
            "population_size": pop_size,
            "mutation_rate": mutation_rate,
            "elite_fraction": elite_fraction
        }
        
        return best_sample, best_score, history, metadata 