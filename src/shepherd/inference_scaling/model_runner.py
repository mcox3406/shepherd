"""
Model runner for inference-time scaling with ShEPhERD.
"""

import numpy as np
import torch
import logging
import copy
from functools import partial


class ShepherdModelRunner:
    """Interface for running the ShEPhERD model with specific noise vectors."""
    
    def __init__(self, 
                 model_pl, 
                 batch_size=1, 
                 N_x1=25, 
                 N_x4=5, 
                 unconditional=True,
                 device="cpu",
                 save_all_noise=False,
                 sampler_type='ddpm',
                 num_steps=None,
                 ddim_eta=0.0,
                 **inference_kwargs):
        """
        Initialize the ShepherdModelRunner.
        
        Args:
            model_pl: PyTorch Lightning module for the ShEPhERD model.
            batch_size (int): Batch size for inference.
            N_x1 (int): Number of atoms to generate.
            N_x4 (int): Number of pharmacophores to generate.
            unconditional (bool): Whether to run unconditional generation.
            device (str): Device to run the model on ("cpu" or "cuda").
            save_all_noise (bool): Whether to save all noise vectors used.
            **inference_kwargs: Additional keyword arguments for inference_sample.
        """
        self.model_pl = model_pl
        self.batch_size = batch_size
        self.N_x1 = N_x1
        self.N_x4 = N_x4
        self.unconditional = unconditional
        self.device = device
        self.inference_kwargs = inference_kwargs
        self.save_all_noise = save_all_noise
        
        # store sampler parameters
        self.sampler_type = sampler_type
        self.num_steps = num_steps
        self.ddim_eta = ddim_eta
        
        # store the last noise used for inference
        self.last_noise = None
        self.all_noise = [] if save_all_noise else None
        
        # function to run inference
        self.inference_function = partial(
            self._run_inference_sample,
            model_pl=self.model_pl,
            batch_size=self.batch_size,
            N_x1=self.N_x1,
            N_x4=self.N_x4,
            unconditional=self.unconditional,
            sampler_type=self.sampler_type,
            num_steps=self.num_steps,
            ddim_eta=self.ddim_eta,
            **self.inference_kwargs
        )
    
    def __call__(self, noise=None):
        """
        Run inference with the ShEPhERD model.
        
        Args:
            noise (torch.Tensor, optional): Specific noise vector to use. If None,
                                           a random noise vector is used.
                                           
        Returns:
            dict: Generated molecule data.
        """
        if noise is not None:
            self.last_noise = noise
            if self.save_all_noise:
                self.all_noise.append(copy.deepcopy(noise))
        
        # run inference with the ShEPhERD model
        samples = self.inference_function(custom_noise=noise)
        
        # return the first sample (we're currently using batch_size=1)
        return samples[0]
    
    def get_last_noise(self):
        """
        Get the last noise vector used for inference.
        
        Returns:
            torch.Tensor: Last noise vector used.
        """
        return self.last_noise
    
    def get_all_noise(self):
        """
        Get all noise vectors used for inference.
        
        Returns:
            list: All noise vectors used.
        """
        if not self.save_all_noise:
            logging.warning("save_all_noise was set to False. No noise vectors were saved.")
        return self.all_noise
    
    def _run_inference_sample(self, model_pl, batch_size, N_x1, N_x4, 
                             unconditional, custom_noise=None, **kwargs):
        """
        Run inference with the ShEPhERD model.
        
        This is a wrapper around inference_sample that allows us to use custom
        noise vectors. It hooks into the ShEPhERD inference process to replace
        the random noise with our custom noise.
        
        Args:
            model_pl: PyTorch Lightning module for the ShEPhERD model.
            batch_size (int): Batch size for inference.
            N_x1 (int): Number of atoms to generate.
            N_x4 (int): Number of pharmacophores to generate.
            unconditional (bool): Whether to run unconditional generation.
            custom_noise (torch.Tensor, optional): Custom noise vector to use.
            **kwargs: Additional keyword arguments for inference_sample.
            
        Returns:
            list: Generated molecule data.
        """
        from shepherd.inference import inference_sample
        
        # save the original torch.randn function
        original_randn = torch.randn
        
        # create a list to store the noise vectors
        noise_vectors = []
        
        # define a custom randn function
        def custom_randn(*args, **kwargs):
            if custom_noise is not None and len(noise_vectors) == 0:
                # return our custom noise for the first call
                noise_vectors.append(custom_noise)
                return custom_noise
            else:
                # for subsequent calls, use the original randn function
                noise = original_randn(*args, **kwargs)
                noise_vectors.append(noise)
                return noise
        
        try:
            # replace torch.randn with our custom function
            torch.randn = custom_randn
            
            # run inference
            samples = inference_sample(
                model_pl=model_pl,
                batch_size=batch_size,
                N_x1=N_x1,
                N_x4=N_x4,
                unconditional=unconditional,
                **kwargs
            )
            
            # store the noise vectors
            if len(noise_vectors) > 0:
                self.last_noise = noise_vectors[0]
            
            return samples
        
        finally:
            # restore the original torch.randn function
            torch.randn = original_randn 