"""
Search algorithms for inference-time scaling with ShEPhERD.
"""

import numpy as np
import torch
import logging
import time
from tqdm import tqdm
from copy import deepcopy
from functools import partial
import sys

from rdkit import Chem

from shepherd.inference import inference_sample
from shepherd.inference.initialization import (
    _initialize_x1_state, _initialize_x2_state, _initialize_x3_state, _initialize_x4_state
)
from shepherd.inference.noise import forward_jump, _get_noise_params_for_timestep
from shepherd.inference.steps import (
    _inference_step,
    _prepare_model_input,
    _perform_reverse_denoising_step,
    _extract_generated_samples
)


# Context manager to temporarily modify the root logger's level
class _ModifyRootLoggerLevelContext:
    def __init__(self, temp_level=logging.CRITICAL):
        self.root_logger = logging.getLogger() # Get the root logger
        self.original_level = None
        self.temp_level = temp_level

    def __enter__(self):
        self.original_level = self.root_logger.level
        self.root_logger.setLevel(self.temp_level)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_level is not None:
            self.root_logger.setLevel(self.original_level)


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
        
    def search(self, num_trials=50, verbose=True, device="cpu", callback=None):
        """
        Run random search to find the best noise vector.
        
        Args:
            num_trials (int): Number of random noise vectors to try.
            verbose (bool): Whether to print progress information.
            device (str): Device to run the model on ("cpu" or "cuda"). 
                          Note: for RandomSearch using direct inference_sample, 
                          model_pl should already be on the correct device.
            callback (callable, optional): Function to call after each trial with signature
                                        callback(trial, sample, score, best_sample, best_score, scores).
            
        Returns:
            tuple: (best_sample, best_score, scores_history, metadata)
        """
        best_score = -float('inf')
        best_sample = None
        scores = []
        times = []
        nfe_count = 0
        
        # Calculate number of batches needed to reach num_trials
        num_batches = int(np.ceil(num_trials / self.model_runner.batch_size))
        # if verbose:
        logging.info(f"Running {num_batches} batches of size {self.model_runner.batch_size} to achieve {num_trials} trials")
        
        if verbose:
            pbar = tqdm(total=num_batches, desc="Random Search", position=0, leave=True)
        else:
            pbar = None

        processed_trials = 0
        while processed_trials < num_trials:
            batch_start_time = time.time()
            
            # Directly call inference_sample for a batch of random samples
            # Arguments are sourced from self.model_runner attributes
            samples_in_batch = inference_sample(
                model_pl=self.model_runner.model_pl,
                batch_size=self.model_runner.batch_size,
                N_x1=self.model_runner.N_x1,
                N_x4=self.model_runner.N_x4,
                unconditional=self.model_runner.unconditional,
                sampler_type=self.model_runner.sampler_type,
                num_steps=self.model_runner.num_steps,
                ddim_eta=self.model_runner.ddim_eta,
                verbose=False,
                **self.model_runner.inference_kwargs 
            )
            
            if not isinstance(samples_in_batch, list):
                logging.error("inference_sample did not return a list. Wrapping in a list.")
                samples_in_batch = [samples_in_batch]

            for sample_idx_in_batch, sample in enumerate(samples_in_batch):
                if processed_trials >= num_trials:
                    break
                
                current_sample_score = self.verifier(sample)
                nfe_count += 1
                scores.append(current_sample_score)
                
                if current_sample_score > best_score:
                    best_score = current_sample_score
                    best_sample = sample 
                    if verbose and pbar:
                        pbar.set_description(f"Random Search (Best: {best_score:.4f})")
                
                if callback is not None:
                    callback(algorithm='random', iteration=processed_trials, sample=sample, score=current_sample_score)
                
                processed_trials += 1

            pbar.update(1)
            pbar.refresh()  # Force refresh the display
            
            times.append(time.time() - batch_start_time) 
        
        if pbar:
            pbar.close()
            
        metadata = {
            "mean_time_per_trial": np.mean(times) / self.model_runner.batch_size if self.model_runner.batch_size > 0 and len(times) > 0 else 0, # Approximate per trial
            "total_time": sum(times),
            "num_trials": num_trials,
            "nfe": nfe_count,
            "batch_size_used": self.model_runner.batch_size
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
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Only add a handler if the root logger doesn't have any handlers
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            # Add a console handler if none exists
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(console_handler)
        
    def search(self, num_steps=20, num_neighbors=5, step_size=0.1, 
               init_noise=None, verbose=True, device="cpu", callback=None):
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
            callback (callable, optional): Function to call after each step with signature
                                        callback(step, pivot_sample, pivot_score, best_sample, best_score, scores).
            
        Returns:
            tuple: (best_sample, best_score, scores_history, metadata)
        """
        nfe_count = 0
        if init_noise is None:
            # get a random noise vector from the model_runner
            init_sample = self.model_runner()
            init_score = self.verifier(init_sample)
            nfe_count += 1
            pivot_noise = self.model_runner.get_last_noise()
            pivot_sample = init_sample
            pivot_score = init_score
        else:
            pivot_noise = init_noise
            pivot_sample = self.model_runner(noise=pivot_noise)
            pivot_score = self.verifier(pivot_sample)
            nfe_count += 1
        
        # track best sample and score across all iterations
        best_sample = pivot_sample
        best_score = pivot_score
        
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
            neighbor_noises = []
            
            for n in range(num_neighbors):
                # add random perturbation to the pivot noise
                perturbation = torch.randn_like(pivot_noise) * step_size
                neighbor_noise = pivot_noise + perturbation
                
                # generate sample with the perturbed noise
                neighbor_sample = self.model_runner(noise=neighbor_noise)
                
                # evaluate sample
                with _ModifyRootLoggerLevelContext(temp_level=logging.CRITICAL):
                    neighbor_score = self.verifier(neighbor_sample)
                nfe_count += 1
                
                neighbor_samples.append(neighbor_sample)
                neighbor_scores.append(neighbor_score)
                neighbor_noises.append(neighbor_noise)
                
                # call callback for this specific neighbor evaluation
                if callback is not None:
                    callback(algorithm='zero_order', iteration=step, sub_iteration=n, 
                             sample=neighbor_sample, score=neighbor_score)
                
                # update best overall if this neighbor is better
                if neighbor_score > best_score:
                    best_score = neighbor_score
                    best_sample = neighbor_sample
            
            # find the best neighbor
            best_idx = np.argmax(neighbor_scores)
            best_neighbor_score = neighbor_scores[best_idx]
            
            # update pivot if the best neighbor is better
            if best_neighbor_score > pivot_score:
                pivot_sample = neighbor_samples[best_idx]
                pivot_score = best_neighbor_score
                pivot_noise = neighbor_noises[best_idx]
                
                if verbose:
                    self.logger.info(f"Step {step+1}/{num_steps}: New pivot score: {pivot_score:.4f}")
            
            scores.append(pivot_score)
            times.append(time.time() - start_time)
            
            # if callback is not None:
            #     callback(step, pivot_sample, pivot_score, best_sample, best_score, scores)
        
        metadata = {
            "mean_time_per_step": np.mean(times),
            "total_time": sum(times),
            "num_steps": num_steps,
            "num_neighbors": num_neighbors,
            "step_size": step_size,
            "nfe": nfe_count
        }
        
        return best_sample, best_score, scores, metadata


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
               elite_fraction=0.2, verbose=True, device="cpu", callback=None):
        """
        Run guided search to find the best noise vector.
        
        Args:
            pop_size (int): Number of noise vectors in the population.
            num_generations (int): Number of generations to evolve.
            mutation_rate (float): Rate of mutation for generating new candidates.
            elite_fraction (float): Fraction of the population to keep as elite.
            verbose (bool): Whether to print progress information.
            device (str): Device to run the model on ("cpu" or "cuda").
            callback (callable, optional): Function to call after each generation with signature
                                        callback(gen, population, best_sample, best_score, history).
            
        Returns:
            tuple: (best_sample, best_score, scores_history, metadata)
        """
        nfe_count = 0
        # initialize population
        population = []
        scores = []
        
        if verbose:
            print("Initializing population...")
        
        for i in range(pop_size):
            sample = self.model_runner()
            score = self.verifier(sample)
            nfe_count += 1
            noise = self.model_runner.get_last_noise()
            
            population.append((sample, noise, score))
            scores.append(score)
        
        # sort by score (descending)
        population.sort(key=lambda x: x[2], reverse=True)
        best_sample, best_noise, best_score = population[0]
        
        history = [best_score]
        times = []
        
        elite_count = max(1, int(pop_size * elite_fraction))
        
        # store initial population via callback
        if callback is not None:
            for idx, (sample, noise, score) in enumerate(population):
                callback(algorithm='guided', iteration=0, sub_iteration=idx,
                         sample=sample, score=score, is_initial=True)

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
                nfe_count += 1
                new_noise_saved = self.model_runner.get_last_noise()
                
                new_population.append((new_sample, new_noise_saved, new_score))

                # call callback for this specific new individual
                if callback is not None:
                    callback(algorithm='guided', iteration=gen + 1, sub_iteration=len(new_population) - 1,
                             sample=new_sample, score=new_score, is_elite=False)
            
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
            
            # if callback is not None:
            #     callback(gen, population, best_sample, best_score, history)
        
        metadata = {
            "mean_time_per_generation": np.mean(times),
            "total_time": sum(times),
            "num_generations": num_generations,
            "population_size": pop_size,
            "mutation_rate": mutation_rate,
            "elite_fraction": elite_fraction,
            "nfe": nfe_count
        }
        
        return best_sample, best_score, history, metadata


class SearchOverPaths(SearchAlgorithm):
    """
    Search Over Paths algorithm for inference-time scaling.

    Iteratively refines N diffusion trajectories by:
    1. Starting N paths by running ODE to an initial sigma.
    2. In each iteration, for each path:
        a. Perturb forward (forward noise by delta_f).
        b. Solve ODE backward by delta_b.
        c. Fully denoise the result for verification.
    3. Keep top N paths based on verifier scores.
    4. Optionally, refine final N paths with a random search.
    """

    def __init__(self, verifier, model_runner, name="SearchOverPaths"):
        """
        Initialize a SearchOverPaths algorithm.

        Args:
            verifier: A verifier callable that evaluates molecular quality.
            model_runner: An object (e.g., InferenceMaeve instance) that provides
                          access to the model (model_pl) and its parameters (params),
                          and can execute inference.
            name (str, optional): Name of the search algorithm.
        """
        super().__init__(verifier, model_runner, name)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Only add a handler if the root logger doesn't have any handlers
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            # Add a console handler if none exists
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(console_handler)

    def _run_diffusion_segment(self,
        initial_states_batch,
        t_start_idx, t_end_idx, time_steps,
        noise_dict,
        batch_size,
        verbose=False):
        """
        Helper to run a segment of the diffusion process.
        Manages iterative calls to _inference_step.
        `initial_states_batch` is a dict of tensors, already batched.
        Returns the state dict at t_end_val.
        """
        current_states = deepcopy(initial_states_batch)

        inference_step = partial(
            _inference_step,
            model_pl=self.model_runner.model_pl,
            params=self.model_runner.model_pl.params,
            time_steps=time_steps,
            harmonize=False, harmonize_ts=[], harmonize_jumps=[],
            batch_size=batch_size,
            sampler_type=self.model_runner.sampler_type, ddim_eta=self.model_runner.ddim_eta,
            denoising_noise_scale=1.0,
            inject_noise_at_ts=self.model_runner.inference_kwargs.get('inject_noise_at_ts', []),
            inject_noise_scales=self.model_runner.inference_kwargs.get('inject_noise_scales', []),
            virtual_node_mask_x1=current_states['virtual_node_mask_x1'],
            virtual_node_mask_x2=current_states['virtual_node_mask_x2'],
            virtual_node_mask_x3=current_states['virtual_node_mask_x3'],
            virtual_node_mask_x4=current_states['virtual_node_mask_x4'],
            # inpainting
            inpaint_x2_pos=False, inpaint_x3_pos=False, inpaint_x3_x=False,
            inpaint_x4_pos=False, inpaint_x4_direction=False,
            inpaint_x4_type=False,
            inpainting_dict=None,
        )

        x1_pos_t = current_states['x1_pos_t']
        x1_x_t=current_states['x1_x_t']
        x1_bond_edge_x_t = current_states['x1_bond_edge_x_t']
        x1_batch = current_states['x1_batch']
        bond_edge_index_x1 = current_states['bond_edge_index_x1']
        x2_pos_t = current_states['x2_pos_t']
        x2_x_t=current_states['x2_x_t']
        x2_batch = current_states['x2_batch']
        x3_pos_t = current_states['x3_pos_t']
        x3_x_t=current_states['x3_x_t']
        x3_batch = current_states['x3_batch']
        x4_pos_t = current_states['x4_pos_t']
        x4_direction_t=current_states['x4_direction_t']
        x4_x_t = current_states['x4_x_t']
        x4_batch = current_states['x4_batch']

        t_end_idx = min(t_end_idx, len(time_steps) - 1)

        if verbose:
            # Configure tqdm to write to stdout instead of stderr
            # Ensure total is non-negative for tqdm
            pbar_total = max(0, t_end_idx - t_start_idx)
            pbar = tqdm(total=pbar_total, position=0, leave=True, miniters=25, mininterval=1000)
            if pbar_total > 0:
                pbar.set_description(f"Simulating SDE: t={t_start_idx} -> t={t_end_idx}")
            else:
                pbar.set_description(f"Simulating SDE: no steps (t={t_start_idx})")
        else:
            pbar = None

        # current_time_idx is already initialized to t_start_idx from the arguments
        # This variable will hold the output of the last inference_step, or be constructed if loop doesn't run
        processed_output_state = None
        # Thread noise_dict through the loop
        _current_noise_dict = deepcopy(noise_dict)

        _last_inference_step_result = None # To store tuple (time_idx, state_dict)

        while t_start_idx < t_end_idx:
            t_start_idx, _one_step_output_dict = inference_step(
                current_time_idx=t_start_idx,
                # current states
                x1_pos_t=x1_pos_t, x1_x_t=x1_x_t, x1_bond_edge_x_t=x1_bond_edge_x_t,
                x1_batch=x1_batch, bond_edge_index_x1=bond_edge_index_x1,
                x2_pos_t=x2_pos_t, x2_x_t=x2_x_t,
                x2_batch=x2_batch,
                x3_pos_t=x3_pos_t, x3_x_t=x3_x_t,
                x3_batch=x3_batch,
                x4_pos_t=x4_pos_t, x4_direction_t=x4_direction_t, x4_x_t=x4_x_t,
                x4_batch=x4_batch,
                # noise
                noise_dict=_current_noise_dict,
                # progress bar
                pbar=pbar
            )
            _last_inference_step_result = (t_start_idx, _one_step_output_dict)


            # update states for the next iteration
            x1_pos_t = _one_step_output_dict['x1_pos_t_1']
            x1_x_t = _one_step_output_dict['x1_x_t_1']
            x1_bond_edge_x_t = _one_step_output_dict['x1_bond_edge_x_t_1']
            x2_pos_t = _one_step_output_dict['x2_pos_t_1']
            x2_x_t = _one_step_output_dict['x2_x_t_1']
            x3_pos_t = _one_step_output_dict['x3_pos_t_1']
            x3_x_t = _one_step_output_dict['x3_x_t_1']
            x4_pos_t = _one_step_output_dict['x4_pos_t_1']
            x4_direction_t = _one_step_output_dict['x4_direction_t_1']
            x4_x_t = _one_step_output_dict['x4_x_t_1']
            _current_noise_dict = _one_step_output_dict['noise_dict']
        
        if verbose and (pbar is not None): # Ensure pbar exists before closing
            pbar.close()

        if _last_inference_step_result is not None: # Loop ran at least once
            # current_time_idx is already updated from the loop
            processed_output_state = deepcopy(_last_inference_step_result[1]) # This is the state dict with _t_1 keys
        else: # Loop was skipped, current_time_idx is still t_start_idx
            processed_output_state = {}
            # Construct state from current_states (deepcopy of initial_states_batch), renaming _t to _t_1
            for key, value in current_states.items():
                if '_t' in key and not key.endswith('_t_1') and isinstance(value, torch.Tensor): # Check if it's a tensor that needs renaming
                    new_key = key.replace('_t', '_t_1')
                    processed_output_state[new_key] = value
                elif key in ['noise_dict', 'bond_edge_index_x1', 
                             'virtual_node_mask_x1', 'virtual_node_mask_x2', 
                             'virtual_node_mask_x3', 'virtual_node_mask_x4',
                             'x1_batch', 'x2_batch', 'x3_batch', 'x4_batch']:
                    # Carry over essential non-time-evolved properties
                    if key == 'noise_dict':
                        processed_output_state[key] = _current_noise_dict # Use the initial noise_dict if loop didn't run
                    else:
                        processed_output_state[key] = value
            # Ensure all expected _t_1 keys are present if not converted, e.g. if initial_states_batch already had them
            # This block primarily handles initial_states_batch having _t keys.

        # Add/overwrite with static information from current_states (initial state of segment)
        # These are the assignments that previously caused UnboundLocalError if next_state wasn't set
        processed_output_state['bond_edge_index_x1'] = current_states['bond_edge_index_x1']
        processed_output_state['virtual_node_mask_x1'] = current_states['virtual_node_mask_x1']
        processed_output_state['virtual_node_mask_x2'] = current_states['virtual_node_mask_x2']
        processed_output_state['virtual_node_mask_x3'] = current_states['virtual_node_mask_x3']
        processed_output_state['virtual_node_mask_x4'] = current_states['virtual_node_mask_x4']
        # User's additions for batch tensors
        processed_output_state['x1_batch'] = current_states['x1_batch']
        processed_output_state['x2_batch'] = current_states['x2_batch']
        processed_output_state['x3_batch'] = current_states['x3_batch']
        processed_output_state['x4_batch'] = current_states['x4_batch']
        
        # current_time_idx is the time at the end of the segment (t_end_idx if loop ran, or t_start_idx if skipped)
        return t_start_idx, processed_output_state


    def _apply_forward_jump_to_state_batch(self, current_states, current_time_idx, jump_steps, time_steps, M_expansion_per_path, N_x1, N_x4, verbose):
        """
        Assumes that this is a batch size of 1.
        Forward jump ∆f steps (jump_steps).

        Returns a new state_dict that has a batch size of M_expansion_per_path
        """
        t_start_val = time_steps[current_time_idx]
        if jump_steps > self.model_runner.model_pl.params['noise_schedules']['x1']['ts'].max() - t_start_val:
            jump_steps = self.model_runner.model_pl.params['noise_schedules']['x1']['ts'].max() - t_start_val

        noised_states = deepcopy(current_states)

        if verbose:
            self.logger.info(f"Forward jump: t={current_time_idx} -> t={current_time_idx -jump_steps}")

        x1_pos_t = noised_states['x1_pos_t_1']
        x1_x_t = noised_states['x1_x_t_1']
        x1_bond_edge_x_t = noised_states['x1_bond_edge_x_t_1']
        nbonds = x1_bond_edge_x_t.shape[0]
        # x1_batch = noised_states['x1_batch']
        bond_edge_index_x1 = noised_states['bond_edge_index_x1']
        x2_pos_t = noised_states['x2_pos_t_1']
        x2_x_t = noised_states['x2_x_t_1']
        # x2_batch = noised_states['x2_batch']
        x3_pos_t = noised_states['x3_pos_t_1']
        x3_x_t = noised_states['x3_x_t_1']
        # x3_batch = noised_states['x3_batch']
        x4_pos_t = noised_states['x4_pos_t_1']
        x4_direction_t = noised_states['x4_direction_t_1']
        x4_x_t = noised_states['x4_x_t_1']
        # x4_batch = noised_states['x4_batch']

        virtual_node_mask_x1 = noised_states['virtual_node_mask_x1']
        virtual_node_mask_x2 = noised_states['virtual_node_mask_x2']
        virtual_node_mask_x3 = noised_states['virtual_node_mask_x3']
        virtual_node_mask_x4 = noised_states['virtual_node_mask_x4']

        # Create batch indices before concatenating tensors
        x1_batch = torch.arange(M_expansion_per_path).repeat_interleave(N_x1 + 1)
        x2_batch = torch.arange(M_expansion_per_path).repeat_interleave(75 + 1)
        x3_batch = torch.arange(M_expansion_per_path).repeat_interleave(75 + 1)
        x4_batch = torch.arange(M_expansion_per_path).repeat_interleave(N_x4 + 1)

        # Properly handle edge indices for batched data
        offset_edge_indices = []
        for i in range(M_expansion_per_path):
            # First offset by the number of bonds in previous batches
            # batch_edges = bond_edge_index_x1 + i*nbonds
            # Then offset by the number of nodes in previous batches to account for virtual node
            batch_edges = bond_edge_index_x1 + i*(N_x1 + 1)
            offset_edge_indices.append(batch_edges)
        batched_edge_index = torch.cat(offset_edge_indices, dim=1)
        
        x1_sigma_ts = self.model_runner.model_pl.params['noise_schedules']['x1']['sigma_ts']
        x2_sigma_ts = self.model_runner.model_pl.params['noise_schedules']['x2']['sigma_ts']
        x3_sigma_ts = self.model_runner.model_pl.params['noise_schedules']['x3']['sigma_ts']
        x4_sigma_ts = self.model_runner.model_pl.params['noise_schedules']['x4']['sigma_ts']

        virtual_node_mask_x1 = torch.cat([virtual_node_mask_x1]*M_expansion_per_path, dim=0)
        virtual_node_mask_x2 = torch.cat([virtual_node_mask_x2]*M_expansion_per_path, dim=0)
        virtual_node_mask_x3 = torch.cat([virtual_node_mask_x3]*M_expansion_per_path, dim=0)
        virtual_node_mask_x4 = torch.cat([virtual_node_mask_x4]*M_expansion_per_path, dim=0)
        
        # Repeat / batch so that we have M different new paths
        # print(f't_start_val: {type(t_start_val)}')
        # print(f'jump_steps: {type(jump_steps)}')
        x1_pos_t, x1_t_jump = forward_jump(torch.cat([x1_pos_t]*M_expansion_per_path, dim=0), t_start_val, jump_steps, x1_sigma_ts, remove_COM_from_noise = True, batch = x1_batch, mask = ~virtual_node_mask_x1)
        x1_x_t, _ = forward_jump(torch.cat([x1_x_t]*M_expansion_per_path, dim=0), t_start_val, jump_steps, x1_sigma_ts, remove_COM_from_noise = False, batch = x1_batch, mask = ~virtual_node_mask_x1)
        x1_bond_edge_x_t, _ = forward_jump(torch.cat([x1_bond_edge_x_t]*M_expansion_per_path, dim=0), t_start_val, jump_steps, x1_sigma_ts, remove_COM_from_noise = False, batch = None, mask = None)
        
        x2_pos_t, _ = forward_jump(torch.cat([x2_pos_t]*M_expansion_per_path, dim=0), t_start_val, jump_steps, x2_sigma_ts, remove_COM_from_noise = False, batch = x2_batch, mask = ~virtual_node_mask_x2)
        x2_x_t, _ = forward_jump(torch.cat([x2_x_t]*M_expansion_per_path, dim=0), t_start_val, jump_steps, x2_sigma_ts, remove_COM_from_noise = False, batch = x2_batch, mask = ~virtual_node_mask_x2)
        
        x3_pos_t, _ = forward_jump(torch.cat([x3_pos_t]*M_expansion_per_path, dim=0), t_start_val, jump_steps, x3_sigma_ts, remove_COM_from_noise = False, batch = x3_batch, mask = ~virtual_node_mask_x3)
        x3_x_t, _ = forward_jump(torch.cat([x3_x_t]*M_expansion_per_path, dim=0), t_start_val, jump_steps, x3_sigma_ts, remove_COM_from_noise = False, batch = x3_batch, mask = ~virtual_node_mask_x3)
        
        x4_pos_t, _ = forward_jump(torch.cat([x4_pos_t]*M_expansion_per_path, dim=0), t_start_val, jump_steps, x4_sigma_ts, remove_COM_from_noise = False, batch = x4_batch, mask = ~virtual_node_mask_x4)
        x4_direction_t, _ = forward_jump(torch.cat([x4_direction_t]*M_expansion_per_path, dim=0), t_start_val, jump_steps, x4_sigma_ts, remove_COM_from_noise = False, batch = x4_batch, mask = ~virtual_node_mask_x4)
        x4_x_t, _ = forward_jump(torch.cat([x4_x_t]*M_expansion_per_path, dim=0), t_start_val, jump_steps, x4_sigma_ts, remove_COM_from_noise = False, batch = x4_batch, mask = ~virtual_node_mask_x4)

        jumped_to_t = x1_t_jump # assuming all jumps are same length
        # find where the jumped-to time occurs in the sequence
        jump_to_idx = np.where(time_steps == jumped_to_t)[0][0]

        noised_states = {}
        noised_states['x1_pos_t'] = x1_pos_t
        noised_states['x1_x_t'] = x1_x_t
        noised_states['x1_bond_edge_x_t'] = x1_bond_edge_x_t
        # noised_states['bond_edge_index_x1'] = torch.cat([bond_edge_index_x1 + i*nbonds for i in range(M_expansion_per_path)], dim=1)
        noised_states['bond_edge_index_x1'] = batched_edge_index
        noised_states['x2_pos_t'] = x2_pos_t
        noised_states['x2_x_t'] = x2_x_t
        noised_states['x3_pos_t'] = x3_pos_t
        noised_states['x3_x_t'] = x3_x_t
        noised_states['x4_pos_t'] = x4_pos_t
        noised_states['x4_direction_t'] = x4_direction_t
        noised_states['x4_x_t'] = x4_x_t
        noised_states['x1_batch'] = x1_batch
        noised_states['x2_batch'] = x2_batch
        noised_states['x3_batch'] = x3_batch
        noised_states['x4_batch'] = x4_batch
        noised_states['virtual_node_mask_x1'] = virtual_node_mask_x1
        noised_states['virtual_node_mask_x2'] = virtual_node_mask_x2
        noised_states['virtual_node_mask_x3'] = virtual_node_mask_x3
        noised_states['virtual_node_mask_x4'] = virtual_node_mask_x4


        return jump_to_idx, noised_states


    def _convert_state_to_verifiable_format(self, state_at_t, current_time_idx, time_steps, batch_size):
        """
        Converts an internal state dict (at t=t) to the format expected by the verifier.
        This logic is similar to the end of `inference_sample`.
        This is a simplified placeholder.
        """
        new_state_at_t = {}
        for key, value in state_at_t.items():
            if '_t_1' in key:
                new_state_at_t[key.replace('_t_1', '_t')] = value
            else:
                new_state_at_t[key] = value

        _, next_state = self._run_diffusion_segment(
            new_state_at_t,
            current_time_idx, len(time_steps) - 1, time_steps,
            None,
            batch_size,
            verbose=False
        )
        x1_pos_t = next_state['x1_pos_t_1']
        x1_x_t = next_state['x1_x_t_1']
        x1_bond_edge_x_t = next_state['x1_bond_edge_x_t_1']
        x2_pos_t = next_state['x2_pos_t_1']
        x2_x_t = next_state['x2_x_t_1']
        x3_pos_t = next_state['x3_pos_t_1']
        x3_x_t = next_state['x3_x_t_1']
        x4_pos_t = next_state['x4_pos_t_1']
        x4_direction_t = next_state['x4_direction_t_1']
        x4_x_t = next_state['x4_x_t_1']

        virtual_node_mask_x1 = next_state['virtual_node_mask_x1']
        virtual_node_mask_x2 = next_state['virtual_node_mask_x2']
        virtual_node_mask_x3 = next_state['virtual_node_mask_x3']
        virtual_node_mask_x4 = next_state['virtual_node_mask_x4']

        generated_structures = _extract_generated_samples(
            x1_x_t, x1_pos_t, x1_bond_edge_x_t, virtual_node_mask_x1,
            x2_pos_t, virtual_node_mask_x2,
            x3_pos_t, x3_x_t, virtual_node_mask_x3,
            x4_pos_t, x4_direction_t, x4_x_t, virtual_node_mask_x4,
            self.model_runner.model_pl.params, batch_size
        )
        return generated_structures
    

    def _evaluate_with_verifier(self, generated_structures):
        """
        Evaluate the generated structures with the verifier.

        Returns list of scores for each generated structure.
        """
        scores = []
        # Temporarily raise root logger level to CRITICAL to suppress 
        # INFO, WARNING, and ERROR messages from verifiers that use the root logger.
        with _ModifyRootLoggerLevelContext(temp_level=logging.CRITICAL):
            for structure in generated_structures:
                # Evaluate the generated structure with the verifier
                score = self.verifier(structure)
                scores.append(score)
        return scores


    def search(
        self, N_paths, M_expansion_per_path,
        sigma_t_initial, delta_f_steps, delta_b_steps,
        N_x1=40, N_x4=10, unconditional=True, prior_noise_scale=1.0,
        verbose=True, device='cpu', callback=None):
        """
        Search Over Paths algorithm implementation.
        
        Args:
            N_paths (int): Number of paths to initialize.
            M_expansion_per_path (int): Number of expansions per path.
            sigma_t_initial (float): Initial sigma value for the paths.
            delta_f_steps (int): Number of steps for forward noise perturbation.
            delta_b_steps (int): Number of steps for backward denoising.
            N_x1 (int): Number of x1 nodes in the model.
            N_x4 (int): Number of x4 nodes in the model.
            unconditional (bool): Whether to use unconditional generation.
            prior_noise_scale (float): Scale for prior noise.
            verbose (bool): Whether to print progress information.
            device (str): Device to run the model on ("cpu" or "cuda").
            callback (callable, optional): Function to call after each iteration with signature
                                        callback(iteration, sample, score, best_sample, best_score, scores).
        
        Returns:
            tuple: (best_sample, best_score, scores_history, metadata)
        """

        if not (delta_b_steps > delta_f_steps):
            raise ValueError("delta_b_steps must be strictly greater than delta_f_steps.")
        
        params = self.model_runner.model_pl.params
        T = params['noise_schedules']['x1']['ts'].max()
        time_steps = np.arange(T, 0, -1)

        N_x2 = 75
        N_x3 = 75
        
        self.logger.info(f"Starting SearchOverPaths with N={N_paths} paths and M={M_expansion_per_path} samples per path")
        self.logger.info(f"Initial sigma: {sigma_t_initial}, delta_f: {delta_f_steps}, delta_b: {delta_b_steps}")

        nfe_per_path = self.nfe(sigma_t_initial, delta_f_steps, delta_b_steps)
        self.logger.info(f"Expected NFE per path: {nfe_per_path}")

        # Initialize for all N paths
        self.logger.info("Initializing paths...")
        
        # Initialize x1 state
        (x1_pos_T, x1_x_T, x1_bond_edge_x_T, 
        x1_batch, virtual_node_mask_x1, bond_edge_index_x1) = _initialize_x1_state(
            N_paths, N_x1, params, prior_noise_scale, True
        )

        # Initialize x2 state
        x2_pos_T, x2_x_T, x2_batch, virtual_node_mask_x2 = _initialize_x2_state(
            N_paths, 75, params, prior_noise_scale, True
        )

        # Initialize x3 state
        x3_pos_T, x3_x_T, x3_batch, virtual_node_mask_x3 = _initialize_x3_state(
            N_paths, 75, params, prior_noise_scale, True
        )

        # Initialize x4 state
        (x4_pos_T, x4_direction_T, x4_x_T, 
        x4_batch, virtual_node_mask_x4) = _initialize_x4_state(
            N_paths, N_x4, params, prior_noise_scale, True
        )

        all_scores_history = {
            "num_paths": N_paths,
            "paths_width": M_expansion_per_path,
            "nfe_per_path": nfe_per_path,
        }

        top_N = {}
        times = []
        nfe_count = 0

        iterator = tqdm(range(N_paths), desc=f"Search Over Paths | N={N_paths} | M={M_expansion_per_path} | L={nfe_per_path}", position=0, leave=True)

        for n_path in iterator:
            start_time = time.time()
            all_scores_history[n_path] = {}

            best_score = 0.0
            best_score_evolution = []

            x1_batch = torch.zeros(N_x1 + 1, dtype = torch.long)
            x2_batch = torch.zeros(N_x2 + 1, dtype = torch.long)
            x3_batch = torch.zeros(N_x3 + 1, dtype = torch.long)
            x4_batch = torch.zeros(N_x4 + 1, dtype = torch.long)
            
            num_bonds = int(x1_bond_edge_x_T.shape[0] / N_paths)
            n_x1 = int(x1_pos_T.shape[0] / N_paths)
            n_x2 = int(x2_pos_T.shape[0] / N_paths)
            n_x3 = int(x3_pos_T.shape[0] / N_paths)
            n_x4 = int(x4_pos_T.shape[0] / N_paths)
            
            path_state = {
                'x1_pos_t': x1_pos_T[n_path*n_x1:(n_path+1)*n_x1,:], # (N_paths*(N_x1+1), 3) -> (N_x1+1, 3)
                'x1_x_t': x1_x_T[n_path*n_x1:(n_path+1)*n_x1,:],     # (N_paths*(N_x1+1), 17) -> (N_x1+1, 17)
                'x1_bond_edge_x_t': x1_bond_edge_x_T[n_path*num_bonds:num_bonds*(n_path+1),:], # (N_paths*num_bonds, 5) -> (num_bonds, 5)
                'x1_batch': x1_batch,
                'virtual_node_mask_x1': virtual_node_mask_x1[:n_x1], # (N_paths*(N_x1+1),)
                # For the edge index, it's the same thing copied over and over, just shifted
                # so for a batch size of 1, then it's just be the first set -> num_bonds
                'bond_edge_index_x1': bond_edge_index_x1[:,:num_bonds], # (2, num_bonds*N_paths) -> (2, num_bonds)
                'x2_pos_t': x2_pos_T[n_path*n_x2:(n_path+1)*n_x2,:], # (N_paths*(N_x2+1), 3) -> (N_x2+1, 3)
                'x2_x_t': x2_x_T[n_path*n_x2:(n_path+1)*n_x2,:], # (N_paths*(N_x2+1), 2) -> (N_x2+1, 2)
                'x2_batch': x2_batch, 'virtual_node_mask_x2': virtual_node_mask_x2[:n_x2],
                'x3_pos_t': x3_pos_T[n_path*n_x3:(n_path+1)*n_x3,:], # (N_paths*(N_x3+1), 3) -> (N_x3+1, 3)
                'x3_x_t': x3_x_T[n_path*n_x3:(n_path+1)*n_x3], # (N_paths*(N_x3+1)) -> (N_x3+1)
                'x3_batch': x3_batch, 'virtual_node_mask_x3': virtual_node_mask_x3[:n_x3],
                'x4_pos_t': x4_pos_T[n_path*n_x4:(n_path+1)*n_x4,:], 'x4_direction_t': x4_direction_T[n_path*n_x4:(n_path+1)*n_x4,:],
                'x4_x_t': x4_x_T[n_path*n_x4:(n_path+1)*n_x4,:], 'x4_batch': x4_batch,
                'virtual_node_mask_x4': virtual_node_mask_x4[:n_x4],
            }
            
            # state at sigma_t_initial
            current_time_idx, current_state = self._run_diffusion_segment(
                initial_states_batch=path_state,
                t_start_idx=0, t_end_idx=deepcopy(sigma_t_initial), time_steps=time_steps,
                noise_dict=None,
                batch_size=1,
                verbose=verbose
            )
            
            if verbose:
                self.logger.info(f'Path {n_path+1}/{N_paths}: Initialized at t={current_time_idx}')
            
            retry_ticker = 0
            ticker = 0

            while current_time_idx < len(time_steps) - 1:
                # Go back to a noisier state with forward noising process for delta_f_steps
                # Create M different new noisy states (i.e., paths)
                noised_time_idx, noised_state = self._apply_forward_jump_to_state_batch(
                    current_states=current_state,
                    current_time_idx=current_time_idx, jump_steps=delta_f_steps, time_steps=time_steps,
                    M_expansion_per_path=M_expansion_per_path, N_x1=N_x1, N_x4=N_x4,
                    verbose=verbose
                )

                # Evolve them to time_idx: initial + delta_f - delta_b
                less_noised_time_idx, less_noised_state = self._run_diffusion_segment(
                    initial_states_batch=noised_state,
                    t_start_idx=noised_time_idx,
                    t_end_idx=(noised_time_idx + delta_b_steps),
                    time_steps=time_steps,
                    noise_dict=None,
                    batch_size=M_expansion_per_path,
                    verbose=verbose
                )

                # Get the structures if we evolve all M paths to t=0 (i.e., fully denoised)
                generated_structures = self._convert_state_to_verifiable_format(
                    state_at_t=less_noised_state,
                    current_time_idx=less_noised_time_idx,
                    time_steps=time_steps, batch_size=M_expansion_per_path
                )

                # Score the generated structures
                scores = self._evaluate_with_verifier(generated_structures)
                nfe_count += len(scores)
                
                if max(scores) == 0.0:
                    if retry_ticker < 2:
                        self.logger.warning(f'Path {n_path+1}: All molecules failed, retrying (attempt {retry_ticker + 1}/3)')
                        retry_ticker += 1
                        continue
                    else:
                        self.logger.warning(f'Path {n_path+1}: All molecules failed after 3 attempts, choosing one randomly')
                ticker += 1
                
                # Store the scores
                all_scores_history[n_path][ticker] = {}
                all_scores_history[n_path][ticker]['scores'] = scores
                all_scores_history[n_path][ticker]['generated_structures'] = generated_structures
                all_scores_history[n_path][ticker]['best_idx'] = np.argmax(scores)
                all_scores_history[n_path][ticker]['best_score'] = max(scores)
                
                if max(scores) != 0.0:
                    best_rdmol = self.verifier.preprocess(generated_structures[np.argmax(scores)])
                    all_scores_history[n_path][ticker]['best_rdmol'] = Chem.MolToMolBlock(best_rdmol)
                    all_scores_history[n_path][ticker]['best_structure_output'] = generated_structures[np.argmax(scores)]
                else:
                    all_scores_history[n_path][ticker]['best_rdmol'] = None
                    all_scores_history[n_path][ticker]['best_structure_output'] = generated_structures[0]

                best_score_evolution.append(max(scores))

                if max(scores) > best_score:
                    best_score = max(scores)
                    # if verbose:
                    self.logger.info(f"Path {n_path+1}: New best score: {best_score:.4f}")

                if callback is not None:
                    callback(
                        algorithm='search_over_paths',
                        iteration=n_path+1, sub_iteration=ticker,
                        sample=all_scores_history[n_path][ticker]['best_structure_output'],
                        best_structure_output=all_scores_history[n_path][ticker]['best_structure_output'],
                        best_score=all_scores_history[n_path][ticker]['best_score'],
                        best_rdmol=all_scores_history[n_path][ticker]['best_rdmol'],
                        scores=scores
                    )

                # Get the best structure from this iteration and make it the new current state
                best_idx = np.argmax(scores)
                best_structure = generated_structures[best_idx]

                # Extract the best sample's states from less_noised_state
                best_state = {
                    'x1_pos_t_1': less_noised_state['x1_pos_t_1'][best_idx*n_x1:(best_idx+1)*n_x1,:],
                    'x1_x_t_1': less_noised_state['x1_x_t_1'][best_idx*n_x1:(best_idx+1)*n_x1,:],
                    'x1_bond_edge_x_t_1': less_noised_state['x1_bond_edge_x_t_1'][best_idx*num_bonds:(best_idx+1)*num_bonds,:],
                    'x1_batch': x1_batch,
                    'virtual_node_mask_x1': virtual_node_mask_x1[:n_x1],
                    'bond_edge_index_x1': bond_edge_index_x1[:,:num_bonds],
                    'x2_pos_t_1': less_noised_state['x2_pos_t_1'][best_idx*n_x2:(best_idx+1)*n_x2,:],
                    'x2_x_t_1': less_noised_state['x2_x_t_1'][best_idx*n_x2:(best_idx+1)*n_x2,:],
                    'x2_batch': x2_batch,
                    'virtual_node_mask_x2': virtual_node_mask_x2[:n_x2],
                    'x3_pos_t_1': less_noised_state['x3_pos_t_1'][best_idx*n_x3:(best_idx+1)*n_x3,:],
                    'x3_x_t_1': less_noised_state['x3_x_t_1'][best_idx*n_x3:(best_idx+1)*n_x3],
                    'x3_batch': x3_batch,
                    'virtual_node_mask_x3': virtual_node_mask_x3[:n_x3],
                    'x4_pos_t_1': less_noised_state['x4_pos_t_1'][best_idx*n_x4:(best_idx+1)*n_x4,:],
                    'x4_direction_t_1': less_noised_state['x4_direction_t_1'][best_idx*n_x4:(best_idx+1)*n_x4,:],
                    'x4_x_t_1': less_noised_state['x4_x_t_1'][best_idx*n_x4:(best_idx+1)*n_x4,:],
                    'x4_batch': x4_batch,
                    'virtual_node_mask_x4': virtual_node_mask_x4[:n_x4],
                }

                retry_ticker = 0
                current_time_idx = deepcopy(less_noised_time_idx)
                current_state = deepcopy(best_state)  # Use the best state as the new current state

                # if verbose:
                self.logger.info(f"Path {n_path+1}/{N_paths} | Iteration {ticker} | t_fwd={current_time_idx}, t_int={noised_time_idx} | Best score: {best_score:.4f}")

            # if verbose:
            evol = ' -> '.join([f"{s:.4f}" for s in best_score_evolution])
            self.logger.info(f'Path {n_path+1}/{N_paths} score evolution: {evol}')

            times.append(time.time() - start_time)

            # Store the best sample after the final iteration
            top_N[n_path] = {
                'final_score': best_score_evolution[-1],
                'best_overall_score': max(best_score_evolution),
                'final_rdmol': all_scores_history[n_path][ticker]['best_rdmol'],
                'best_overall_rdmol': all_scores_history[n_path][np.argmax(best_score_evolution) + 1]['best_rdmol'],
                'final_structure_output': all_scores_history[n_path][ticker]['best_structure_output'],
                'best_structure_output': all_scores_history[n_path][np.argmax(best_score_evolution) + 1]['best_structure_output'],
            }

        best_final_score = max([d['final_score'] for d in list(top_N.values())])
        best_overall_score = max([d['best_overall_score'] for d in list(top_N.values())])
        # get the index of these paths
        best_final_path_idx = np.argmax([d['final_score'] for d in list(top_N.values())])
        best_overall_path_idx = np.argmax([d['best_overall_score'] for d in list(top_N.values())])
        best_final_sample = top_N[best_final_path_idx]['final_structure_output']
        best_overall_sample = top_N[best_overall_path_idx]['best_structure_output']
        best_final_rdmol = top_N[best_final_path_idx]['final_rdmol']
        best_overall_rdmol = top_N[best_overall_path_idx]['best_overall_rdmol']

        metadata = {
            "mean_time_per_paths": np.mean(times),
            "total_time": sum(times),
            "num_paths": N_paths,
            "paths_width": M_expansion_per_path,
            "nfe": nfe_count,
            "nfe_per_path": nfe_per_path,
            "num_iterations": ticker,
        }
        
        if verbose:
            self.logger.info(f"SearchOverPaths completed:")
            self.logger.info(f"  - Best final score: {best_final_score:.4f}")
            self.logger.info(f"  - Best overall score: {best_overall_score:.4f}")
            self.logger.info(f"  - Total NFE: {nfe_count}")
            self.logger.info(f"  - Average time per path: {np.mean(times):.2f}s")
            self.logger.info(f"  - Total time: {sum(times):.2f}s")
            
        return (
            best_final_score,
            best_overall_score,
            best_final_path_idx,
            best_overall_path_idx,
            best_final_sample,
            best_overall_sample,
            best_final_rdmol,
            best_overall_rdmol,
            top_N,
            all_scores_history,
            metadata
        )

    def nfe(self, sigma_t_initial, delta_f_steps, delta_b_steps):
        """
        Calculate the number of function evaluations (NFE) for the search algorithm.
        
        Args:
            sigma_t_initial (float): Initial sigma value for the paths.
            delta_f_steps (int): Number of steps for forward noise perturbation.
            delta_b_steps (int): Number of steps for backward denoising.
        
        Returns:
            int: The number of function evaluations.
        """
        params = self.model_runner.model_pl.params
        T = params['noise_schedules']['x1']['ts'].max()
        time_steps = np.arange(T, 0, -1)

        current_t = sigma_t_initial
        num_func_evals = 0
        while current_t < len(time_steps) - 1:
            current_t += - delta_f_steps + delta_b_steps
            num_func_evals += 1
        return num_func_evals
