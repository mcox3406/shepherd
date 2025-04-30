import open3d
import rdkit
from rdkit.Chem import rdDetermineBonds
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch_geometric
import torch_scatter
import pickle
from copy import deepcopy
import os
from tqdm import tqdm
import sys

sys.path.insert(-1, "../model/")
sys.path.insert(-1, "../model/equiformer_v2")

import pytorch_lightning as pl
from shepherd.lightning_module import LightningModule
from shepherd.datasets import HeteroDataset

from .initialization import (
    _initialize_x1_state,
    _initialize_x2_state,
    _initialize_x3_state,
    _initialize_x4_state
)
from .noise import (
    forward_trajectory,
    _get_noise_params_for_timestep,
    forward_jump
)
from .steps import (
    _perform_reverse_denoising_step,
    _prepare_model_input
)

from shepherd.shepherd_score_utils.generate_point_cloud import (
    get_atom_coords,
    get_atomic_vdw_radii,
    get_molecular_surface,
    get_electrostatics,
    get_electrostatics_given_point_charges,
)
from shepherd.shepherd_score_utils.pharm_utils.pharmacophore import get_pharmacophores
from shepherd.shepherd_score_utils.conformer_generation import update_mol_coordinates


def inference_sample(
    model_pl,
    batch_size,
    
    N_x1,
    N_x4,
    
    unconditional,
    
    prior_noise_scale = 1.0,
    denoising_noise_scale = 1.0,
    
    inject_noise_at_ts = [],
    inject_noise_scales = [],    
    
    harmonize = False,
    harmonize_ts = [],
    harmonize_jumps = [],
    
    # Sampler controls
    sampler_type='ddpm', # 'ddpm' or 'ddim'
    ddim_eta=0.0,      # Controls stochasticity for DDIM (0=deterministic)
    num_steps=None,     # Number of diffusion steps (defaults to T for DDPM)
    
    # all the below options are only relevant if unconditional is False
    
    inpaint_x2_pos = False,
    inpaint_x3_pos = False,
    inpaint_x3_x = False,
    inpaint_x4_pos = False,
    inpaint_x4_direction = False,
    inpaint_x4_type = False,
    
    stop_inpainting_at_time_x2 = 0.0,
    add_noise_to_inpainted_x2_pos = 0.0,
    
    stop_inpainting_at_time_x3 = 0.0,
    add_noise_to_inpainted_x3_pos = 0.0,
    add_noise_to_inpainted_x3_x = 0.0,
    
    stop_inpainting_at_time_x4 = 0.0,
    add_noise_to_inpainted_x4_pos = 0.0,
    add_noise_to_inpainted_x4_direction = 0.0,
    add_noise_to_inpainted_x4_type = 0.0,
    
    # these are the inpainting targets
    center_of_mass = np.zeros(3),
    surface = np.zeros((75,3)),
    electrostatics = np.zeros(75),
    pharm_types = np.zeros(5, dtype = int),
    pharm_pos = np.zeros((5,3)),
    pharm_direction = np.zeros((5,3)),
    
):
    """
    Runs inference of ShEPhERD to sample `batch_size` number of molecules.

    Arguments
    ---------
    model_pl : PyTorch Lightning module.

    batch_size : int Number of molecules to sample in a single batch.

    N_x1 : int Number of atoms to diffuse.
    N_x4 : int Number of pharmacophores to diffuse.
        If inpainting, can be greater than len(pharm_types) for partial pharmacophore conditioning.

    unconditional : bool to toggle unconditional generation.

    prior_noise_scale : float (default = 1.0) Noise scale of the prior distribution.
    denoising_noise_scale : float (default = 1.0) Noise scale for each denoising step.
    
    inject_noise_at_ts : list[int] (default = []) Time steps to inject extra noise.
    inject_noise_scales : list[int] (default = []) Scale of noise to inject at above time steps.
     
    harmonize : bool (default=False) Whether to use harmonization.
    harmonize_ts : list[int] (default = []) Time steps to to harmonization.
    harmonize_jumps : list[int] (default = []) Length of time to harmonize (in time steps).

    *all the below options are only relevant if unconditional is False*
    inpaint_x2_pos : bool (default=False) Toggle inpainting.
        Note that x2 is implicitly modeled via x3.

    inpaint_x3_pos : bool (default=False)
    inpaint_x3_x : bool (default=False)

    inpaint_x4_pos : bool (default=False)
    inpaint_x4_direction : bool (default=False)
    inpaint_x4_type : bool (default=False)
    
    stop_inpainting_at_time_x2 : float (default = 0.0) Time step to stop inpainting.
        t=0.0 implies that inpainting doesn't stop.
    add_noise_to_inpainted_x2_pos : float (default = 0.0) Scale of noise to add to inpainted
        values.
    
    stop_inpainting_at_time_x3 : float (default = 0.0)
    add_noise_to_inpainted_x3_pos : float (default = 0.0)
    add_noise_to_inpainted_x3_x : float (default = 0.0)
    
    stop_inpainting_at_time_x4 : float (default = 0.0)
    add_noise_to_inpainted_x4_pos : float (default = 0.0)
    add_noise_to_inpainted_x4_direction : float (default = 0.0)
    add_noise_to_inpainted_x4_type : float (default = 0.0)
    
    *these are the inpainting targets*
    center_of_mass : np.ndarray (3,) (default = np.zeros(3)) Must be supplied if target molecule is
        not already centered.
    surface : np.ndarray (75,3) (default = np.zeros((75,3)) Surface point coordinates.
    electrostatics : np.ndarray (75,) (default = np.zeros(75)) Electrostatics at each surface point.
    pharm_types : np.ndarray (<=N_x4,) (default = np.zeros(5, dtype = int)) Pharmacophore types.
    pharm_pos : np.ndarray (<=N_x4,3) (default = np.zeros((5,3))) Pharmacophore positions as
        coordinates.
    pharm_direction : np.ndarray (<=N_x4,3) (default = np.zeros((5,3))) Pharmacophore directions as
        unit vectors.

    Returns
    -------
    generated_structures : List[Dict]
        Output dictionary is structured as:
        'x1': {
                'atoms': np.ndarray (N_x1,) of ints for atomic numbers.
                'bonds': np.ndarray of bond types between every atom pair.
                'positions': np.ndarray (N_x1, 3) Coordinates of atoms.
            },
            'x2': {
                'positions': np.ndarray (75, 3) Coordinates of surface points.
            },
            'x3': {
                'charges': np.ndarray (75, 3) ESP at surface points.
                'positions': np.ndarray (75, 3) Coordinates of surface points.
            },
            'x4': {
                'types': np.ndarray (N_x4,) of ints for pharmacophore types.
                'positions': np.ndarray (N_x4, 3) Coordinates of pharmacophores.
                'directions': np.ndarray (N_x4, 3) Unit vectors of pharmacophores.
            },
        }
    """
    
    params = model_pl.params
    
    T = params['noise_schedules']['x1']['ts'].max()
    if num_steps is None:
        num_steps = T
    if sampler_type == 'ddpm' and num_steps != T:
        print("Warning: num_steps is ignored for DDPM sampler. Using full T steps.")
        num_steps = T

    # Determine timestep sequence
    if num_steps >= T:
        time_steps = np.arange(T, 0, -1) # Full sequence [T, T-1, ..., 1]
    else:
        # Linear spacing for DDIM/Accelerated sampling
        # Ensure T is the first step and 0 is the last step to calculate x_0
        time_steps = np.linspace(T, 0, num_steps + 1).round().astype(int)
        time_steps = np.unique(time_steps)[::-1] # Descending order, unique
        if time_steps[0] != T: time_steps = np.insert(time_steps, 0, T)
        if time_steps[-1] != 0: time_steps = np.append(time_steps, 0)
        print(f"Using {sampler_type.upper()} sampling with {len(time_steps)-1} steps: {time_steps[:5]}...{time_steps[-5:]}")

    N_x2 = params['dataset']['x2']['num_points']
    N_x3 = params['dataset']['x3']['num_points']
    
    
    ####### Defining inpainting targets ########

    do_partial_inpainting = False
    assert len(pharm_direction) == len(pharm_pos) and len(pharm_pos) == len(pharm_types)
    assert N_x4 >= len(pharm_pos)
    if N_x4 > len(pharm_pos):
        do_partial_inpainting = True

    # centering about provided center of mass (of x1)
    surface = surface - center_of_mass
    pharm_pos = pharm_pos - center_of_mass
    
    # adding small noise to pharm_pos to avoid overlapping points (causes error when encoding clean structure)
    pharm_pos = pharm_pos + np.random.randn(*pharm_pos.shape) * 0.01
    
    # accounting for virtual nodes
    surface = np.concatenate([np.array([[0.0, 0.0, 0.0]]), surface], axis = 0) # virtual node
    electrostatics = np.concatenate([np.array([0.0]), electrostatics], axis = 0) # virtual node
    pharm_types = pharm_types + 1 # accounting for virtual node as the zeroeth type
    pharm_types = np.concatenate([np.array([0]), pharm_types], axis = 0) # virtual node
    pharm_pos = np.concatenate([np.array([[0.0, 0.0, 0.0]]), pharm_pos], axis = 0) # virtual node
    pharm_direction = np.concatenate([np.array([[0.0, 0.0, 0.0]]), pharm_direction], axis = 0) # virtual node
    
    # one-hot-encodings
    pharm_types_one_hot = np.zeros((pharm_types.size, params['dataset']['x4']['max_node_types']))
    pharm_types_one_hot[np.arange(pharm_types.size), pharm_types] = 1
    pharm_types = pharm_types_one_hot
    
    # scaling features
    electrostatics = electrostatics * params['dataset']['x3']['scale_node_features']
    pharm_types = pharm_types * params['dataset']['x4']['scale_node_features']
    pharm_direction = pharm_direction * params['dataset']['x4']['scale_vector_features']
    
    # defining inpainting targets
    target_inpaint_x2_pos = torch.as_tensor(surface, dtype = torch.float)
    target_inpaint_x2_mask = torch.zeros(surface.shape[0], dtype = torch.long)
    target_inpaint_x2_mask[0] = 1
    target_inpaint_x2_mask = target_inpaint_x2_mask == 0

    target_inpaint_x3_x = torch.as_tensor(electrostatics, dtype = torch.float)
    target_inpaint_x3_pos = torch.as_tensor(surface, dtype = torch.float)
    target_inpaint_x3_mask = torch.zeros(electrostatics.shape[0], dtype = torch.long)
    target_inpaint_x3_mask[0] = 1
    target_inpaint_x3_mask = target_inpaint_x3_mask == 0

    target_inpaint_x4_x = torch.as_tensor(pharm_types, dtype = torch.float)
    target_inpaint_x4_pos = torch.as_tensor(pharm_pos, dtype = torch.float)
    target_inpaint_x4_direction = torch.as_tensor(pharm_direction, dtype = torch.float)
    target_inpaint_x4_mask = torch.zeros(pharm_types.shape[0], dtype = torch.long)
    target_inpaint_x4_mask[0] = 1
    target_inpaint_x4_mask = target_inpaint_x4_mask == 0
    
    deterministic_inpainting_x1 = False
    deterministic_inpainting_x2 = False
    deterministic_inpainting_x3 = False
    deterministic_inpainting_x4 = False
    
    x2_pos_inpainting_trajectory = forward_trajectory(
        x = target_inpaint_x2_pos,
        
        ts = params['noise_schedules']['x2']['ts'],
        alpha_ts = params['noise_schedules']['x2']['alpha_ts'],
        sigma_ts = params['noise_schedules']['x2']['sigma_ts'],
        remove_COM_from_noise = False,
        mask = target_inpaint_x2_mask,
        deterministic = deterministic_inpainting_x2,
    )
    
    x3_pos_inpainting_trajectory = forward_trajectory(
        x = target_inpaint_x3_pos,
        
        ts = params['noise_schedules']['x3']['ts'],
        alpha_ts = params['noise_schedules']['x3']['alpha_ts'],
        sigma_ts = params['noise_schedules']['x3']['sigma_ts'],
        remove_COM_from_noise = False,
        mask = target_inpaint_x3_mask,
        deterministic = deterministic_inpainting_x3,
    )
    x3_x_inpainting_trajectory = forward_trajectory(
        x = target_inpaint_x3_x,
        
        ts = params['noise_schedules']['x3']['ts'],
        alpha_ts = params['noise_schedules']['x3']['alpha_ts'],
        sigma_ts = params['noise_schedules']['x3']['sigma_ts'],
        remove_COM_from_noise = False,
        mask = target_inpaint_x3_mask,
        deterministic = deterministic_inpainting_x3,
    )
    
    x4_x_inpainting_trajectory = forward_trajectory(
        x = target_inpaint_x4_x,
        
        ts = params['noise_schedules']['x4']['ts'],
        alpha_ts = params['noise_schedules']['x4']['alpha_ts'],
        sigma_ts = params['noise_schedules']['x4']['sigma_ts'],
        remove_COM_from_noise = False,
        mask = target_inpaint_x4_mask,
        deterministic = deterministic_inpainting_x4,
    )
    x4_pos_inpainting_trajectory = forward_trajectory(
        x = target_inpaint_x4_pos,
        
        ts = params['noise_schedules']['x4']['ts'],
        alpha_ts = params['noise_schedules']['x4']['alpha_ts'],
        sigma_ts = params['noise_schedules']['x4']['sigma_ts'],
        remove_COM_from_noise = False,
        mask = target_inpaint_x4_mask,
        deterministic = deterministic_inpainting_x4,
    )
    x4_direction_inpainting_trajectory = forward_trajectory(
        x = target_inpaint_x4_direction,
        
        ts = params['noise_schedules']['x4']['ts'],
        alpha_ts = params['noise_schedules']['x4']['alpha_ts'],
        sigma_ts = params['noise_schedules']['x4']['sigma_ts'],
        remove_COM_from_noise = False,
        mask = target_inpaint_x4_mask,
        deterministic = deterministic_inpainting_x4,
    )

    ####################################

    
    # override conditioning options
    if unconditional:
        inpaint_x2_pos = False
        inpaint_x3_pos = False
        inpaint_x3_x = False
        inpaint_x4_pos = False
        inpaint_x4_direction = False
        inpaint_x4_type = False
        
    stop_inpainting_at_time_x2 = int(T*stop_inpainting_at_time_x2)
    stop_inpainting_at_time_x3 = int(T*stop_inpainting_at_time_x3)
    stop_inpainting_at_time_x4 = int(T*stop_inpainting_at_time_x4)
    
    
    ###########  Initializing states at t=T   ##############
    
    include_virtual_node = True
    num_atom_types = len(params['dataset']['x1']['atom_types']) + len(params['dataset']['x1']['charge_types'])
    num_pharm_types = params['dataset']['x4']['max_node_types'] # needed later for inpainting

    # Initialize x1 state
    (pos_forward_noised_x1, x_forward_noised_x1, bond_edge_x_forward_noised_x1, 
     x1_batch, virtual_node_mask_x1, bond_edge_index_x1) = _initialize_x1_state(
         batch_size, N_x1, params, prior_noise_scale, include_virtual_node
     )
    
    # Initialize x2 state
    pos_forward_noised_x2, x_forward_noised_x2, x2_batch, virtual_node_mask_x2 = _initialize_x2_state(
        batch_size, N_x2, params, prior_noise_scale, include_virtual_node
    )

    # Initialize x3 state
    pos_forward_noised_x3, x_forward_noised_x3, x3_batch, virtual_node_mask_x3 = _initialize_x3_state(
        batch_size, N_x3, params, prior_noise_scale, include_virtual_node
    )

    # Initialize x4 state
    (pos_forward_noised_x4, direction_forward_noised_x4, x_forward_noised_x4, 
     x4_batch, virtual_node_mask_x4) = _initialize_x4_state(
         batch_size, N_x4, params, prior_noise_scale, include_virtual_node
     )


    # renaming variables for consistency
    x1_pos_t = pos_forward_noised_x1
    x1_x_t = x_forward_noised_x1
    x1_bond_edge_x_t = bond_edge_x_forward_noised_x1
    
    x2_pos_t = pos_forward_noised_x2
    x2_x_t = x_forward_noised_x2
    
    x3_pos_t = pos_forward_noised_x3
    x3_x_t = x_forward_noised_x3
    
    x4_pos_t = pos_forward_noised_x4
    x4_direction_t = direction_forward_noised_x4
    x4_x_t = x_forward_noised_x4
    
    
    x1_batch_size_nodes = x1_pos_t.shape[0]
    x2_batch_size_nodes = x2_pos_t.shape[0]
    x3_batch_size_nodes = x3_pos_t.shape[0]
    x4_batch_size_nodes = x4_pos_t.shape[0]
    
    x1_t = params['noise_schedules']['x1']['ts'][::-1][0]
    x2_t = params['noise_schedules']['x2']['ts'][::-1][0]
    x3_t = params['noise_schedules']['x3']['ts'][::-1][0]
    x4_t = params['noise_schedules']['x4']['ts'][::-1][0]
    
    t = x1_t
    assert x1_t == x2_t
    assert x1_t == x3_t
    assert x1_t == x4_t
    
    if (x2_t > stop_inpainting_at_time_x2):
        if inpaint_x2_pos:
            x2_pos_t = torch.cat([x2_pos_inpainting_trajectory[x2_t] for _ in range(batch_size)], dim = 0)        
            noise = torch.randn_like(x2_pos_t)
            noise[virtual_node_mask_x2] = 0.0
            x2_pos_t = x2_pos_t + add_noise_to_inpainted_x2_pos * noise
        
    if (x3_t > stop_inpainting_at_time_x3):
        if inpaint_x3_pos:
            x3_pos_t = torch.cat([x3_pos_inpainting_trajectory[x3_t] for _ in range(batch_size)], dim = 0)        
            noise = torch.randn_like(x3_pos_t)
            noise[virtual_node_mask_x3] = 0.0
            x3_pos_t = x3_pos_t + add_noise_to_inpainted_x3_pos * noise
        if inpaint_x3_x:
            x3_x_t = torch.cat([x3_x_inpainting_trajectory[x3_t] for _ in range(batch_size)], dim = 0)
            noise = torch.randn_like(x3_x_t)
            noise[virtual_node_mask_x3] = 0.0
            x3_x_t = x3_x_t + add_noise_to_inpainted_x3_x * noise
        
    if (x4_t > stop_inpainting_at_time_x4):
        if inpaint_x4_pos:
            # x4_pos_t_inpaint = torch.cat([x4_pos_inpainting_trajectory[x4_t] for _ in range(batch_size)], dim = 0)
            x4_pos_t_inpaint = x4_pos_inpainting_trajectory[x4_t].repeat(batch_size, 1, 1)
            if do_partial_inpainting:
                x4_pos_t = x4_pos_t.reshape(batch_size, -1, 3)
                x4_pos_t[:, :x4_pos_t_inpaint.shape[1]] = x4_pos_t_inpaint
                x4_pos_t = x4_pos_t.reshape(-1, 3)
            else:
                x4_pos_t = x4_pos_t_inpaint.reshape(-1,3)
        if inpaint_x4_direction:
            # x4_direction_t_inpaint = torch.cat([x4_direction_inpainting_trajectory[x4_t] for _ in range(batch_size)], dim = 0)
            x4_direction_t_inpaint = x4_direction_inpainting_trajectory[x4_t].repeat(batch_size, 1, 1)
            if do_partial_inpainting:
                x4_direction_t = x4_direction_t.reshape(batch_size, -1, 3)
                x4_direction_t[:, :x4_direction_t_inpaint.shape[1]] = x4_direction_t_inpaint
                x4_direction_t = x4_direction_t.reshape(-1, 3)
            else:
                x4_direction_t = x4_direction_t_inpaint.reshape(-1,3)
        if inpaint_x4_type:
            # x4_x_t_inpaint = torch.cat([x4_x_inpainting_trajectory[x4_t] for _ in range(batch_size)], dim = 0)
            x4_x_t_inpaint = x4_x_inpainting_trajectory[x4_t].repeat(batch_size, 1, 1)
            if do_partial_inpainting:
                x4_x_t = x4_x_t.reshape(batch_size, -1, num_pharm_types)
                x4_x_t[:, :x4_x_t_inpaint.shape[1]] = x4_x_t_inpaint
                x4_x_t = x4_x_t.reshape(-1, num_pharm_types)
            else:
                x4_x_t = x4_x_t_inpaint.reshape(-1, num_pharm_types)
        
    
    
    ######## Main Denoising Loop #########
    
    pbar = tqdm(total=len(time_steps) -1 + sum(harmonize_jumps) * int(harmonize), position=0, leave=True)
    
    x1_t_x_list = []
    x1_t_bond_edge_x_list = []
    x1_t_pos_list = []
    
    x2_t_pos_list = []
    
    x3_t_pos_list = []
    x3_t_x_list = []
    
    x4_t_pos_list = []
    x4_t_direction_list = []
    x4_t_x_list = []
    
    current_time_idx = 0
    while current_time_idx < len(time_steps) - 1:
        current_t = time_steps[current_time_idx]
        prev_t = time_steps[current_time_idx + 1] # The time we are calculating state FOR
        
        # t passed to helpers/model should be current time
        t = current_t 

        # inputs (these might be redundant now, t is the main driver)
        x1_t = t
        x2_t = t
        x3_t = t
        x4_t = t
        
        # harmonize
        # harmonization needs careful consideration with subsequenced timesteps
        # for now, we only harmonize if t is exactly in harmonize_ts
        # a jump might skip over a harmonize_ts value in DDIM
        # if harmonization is used with DDIM, ensure harmonize_ts align with time_steps
        perform_harmonization_jump = False
        harmonize_jump_len = 0
        if (harmonize) and (len(harmonize_ts) > 0) and (t == harmonize_ts[0]):
            print(f'Harmonizing... at time {t}')
            harmonize_ts.pop(0)
            if len(harmonize_ts) == 0:
                harmonize = False # use up harmonization steps
            harmonize_jump_len = harmonize_jumps.pop(0)
            perform_harmonization_jump = True
            
        if perform_harmonization_jump:
            x1_sigma_ts = params['noise_schedules']['x1']['sigma_ts']
            x2_sigma_ts = params['noise_schedules']['x2']['sigma_ts']
            x3_sigma_ts = params['noise_schedules']['x3']['sigma_ts']
            x4_sigma_ts = params['noise_schedules']['x4']['sigma_ts']
            
            x1_pos_t, x1_t_jump = forward_jump(x1_pos_t, x1_t, harmonize_jump_len, x1_sigma_ts, remove_COM_from_noise = True, batch = x1_batch, mask = ~virtual_node_mask_x1)
            x1_x_t, x1_t_jump = forward_jump(x1_x_t, x1_t, harmonize_jump_len, x1_sigma_ts, remove_COM_from_noise = False, batch = x1_batch, mask = ~virtual_node_mask_x1)
            x1_bond_edge_x_t, x1_t_jump = forward_jump(x1_bond_edge_x_t, x1_t, harmonize_jump_len, x1_sigma_ts, remove_COM_from_noise = False, batch = None, mask = None)
            
            x2_pos_t, x2_t_jump = forward_jump(x2_pos_t, x2_t, harmonize_jump_len, x2_sigma_ts, remove_COM_from_noise = False, batch = x2_batch, mask = ~virtual_node_mask_x2)
            
            x3_pos_t, x3_t_jump = forward_jump(x3_pos_t, x3_t, harmonize_jump_len, x3_sigma_ts, remove_COM_from_noise = False, batch = x3_batch, mask = ~virtual_node_mask_x3)
            x3_x_t, x3_t_jump = forward_jump(x3_x_t, x3_t, harmonize_jump_len, x3_sigma_ts, remove_COM_from_noise = False, batch = x3_batch, mask = ~virtual_node_mask_x3)
            
            x4_pos_t, x4_t_jump = forward_jump(x4_pos_t, x4_t, harmonize_jump_len, x4_sigma_ts, remove_COM_from_noise = False, batch = x4_batch, mask = ~virtual_node_mask_x4)
            x4_direction_t, x4_t_jump = forward_jump(x4_direction_t, x4_t, harmonize_jump_len, x4_sigma_ts, remove_COM_from_noise = False, batch = x4_batch, mask = ~virtual_node_mask_x4)
            x4_x_t, x4_t_jump = forward_jump(x4_x_t, x4_t, harmonize_jump_len, x4_sigma_ts, remove_COM_from_noise = False, batch = x4_batch, mask = ~virtual_node_mask_x4)
    
            # after jumping forward, we need to find the corresponding index in our time_steps
            # this simple implementation assumes the jump lands exactly on a future step in the sequence
            # more robust: find the closest step in the sequence
            jumped_to_t = x1_t_jump # assuming all jumps are same length
            try:
                # find where the jumped-to time occurs in the sequence
                jump_to_idx = np.where(time_steps == jumped_to_t)[0][0]
                # reset the loop index to continue from the jumped-to time
                current_time_idx = jump_to_idx
                pbar.update(harmonize_jump_len) # update progress bar for the jumped steps
                print(f"Harmonization jumped from t={t} to t={jumped_to_t}, resuming loop.")
                t = jumped_to_t # update t for the next iteration start
                # need to re-fetch noise params for the new 't' before proceeding if the loop continued immediately,
                # but we will recalculate at the start of the next iteration anyway
                continue # skip the rest of the current loop iteration (denoising step)
            except IndexError:
                 print(f"Warning: Harmonization jumped from t={t} to t={jumped_to_t}, which is not in the planned time_steps sequence {time_steps}. Stopping Harmonization.")
                 harmonize = False # disable future harmonization if jump is incompatible
                 # continue the loop from the *next* scheduled step after the original t

        # inpainting logic
        if (x2_t > stop_inpainting_at_time_x2) and inpaint_x2_pos:
            x2_pos_t = torch.cat([x2_pos_inpainting_trajectory[x2_t] for _ in range(batch_size)], dim = 0)        
            noise = torch.randn_like(x2_pos_t)
            noise[virtual_node_mask_x2] = 0.0
            x2_pos_t = x2_pos_t + add_noise_to_inpainted_x2_pos * noise
        
        if (x3_t > stop_inpainting_at_time_x3):
            if inpaint_x3_pos:
                x3_pos_t = torch.cat([x3_pos_inpainting_trajectory[x3_t] for _ in range(batch_size)], dim = 0)        
                noise = torch.randn_like(x3_pos_t)
                noise[virtual_node_mask_x3] = 0.0
                x3_pos_t = x3_pos_t + add_noise_to_inpainted_x3_pos * noise
            if inpaint_x3_x:
                x3_x_t = torch.cat([x3_x_inpainting_trajectory[x3_t] for _ in range(batch_size)], dim = 0)
                noise = torch.randn_like(x3_x_t)
                noise[virtual_node_mask_x3] = 0.0
                x3_x_t = x3_x_t + add_noise_to_inpainted_x3_x * noise
        
        if (x4_t > stop_inpainting_at_time_x4):
            if inpaint_x4_pos:
                x4_pos_t_inpaint = torch.cat([x4_pos_inpainting_trajectory[x4_t] for _ in range(batch_size)], dim = 0)
                noise = torch.randn_like(x4_pos_t)
                noise[virtual_node_mask_x4] = 0.0
                if do_partial_inpainting:
                    x4_pos_t_inpaint = x4_pos_t_inpaint.reshape(batch_size, -1, 3)
                    noise = noise.reshape(batch_size, -1, 3)[:, :x4_pos_t_inpaint.shape[1]]

                    x4_pos_t_inpaint = x4_pos_t_inpaint + add_noise_to_inpainted_x4_pos * noise
                    x4_pos_t = x4_pos_t.reshape(batch_size, -1, 3)
                    x4_pos_t[:, :x4_pos_t_inpaint.shape[1]] = x4_pos_t_inpaint
                    x4_pos_t = x4_pos_t.reshape(-1, 3)
                else:
                    x4_pos_t_inpaint = x4_pos_t_inpaint + add_noise_to_inpainted_x4_pos * noise
                    x4_pos_t = x4_pos_t_inpaint

            if inpaint_x4_direction:
                x4_direction_t_inpaint = torch.cat([x4_direction_inpainting_trajectory[x4_t] for _ in range(batch_size)], dim = 0)
                noise = torch.randn_like(x4_direction_t)
                noise[virtual_node_mask_x4] = 0.0
                if do_partial_inpainting:
                    x4_direction_t_inpaint = x4_direction_t_inpaint.reshape(batch_size, -1, 3)
                    noise = noise.reshape(batch_size, -1, 3)[:, :x4_direction_t_inpaint.shape[1]]

                    x4_direction_t_inpaint = x4_direction_t_inpaint + add_noise_to_inpainted_x4_direction * noise
                    x4_direction_t = x4_direction_t.reshape(batch_size, -1, 3)
                    x4_direction_t[:, :x4_direction_t_inpaint.shape[1]] = x4_direction_t_inpaint
                    x4_direction_t = x4_direction_t.reshape(-1, 3)
                else:
                    x4_direction_t_inpaint = x4_direction_t_inpaint + add_noise_to_inpainted_x4_direction * noise
                    x4_direction_t = x4_direction_t_inpaint
            if inpaint_x4_type:
                x4_x_t_inpaint = torch.cat([x4_x_inpainting_trajectory[x4_t] for _ in range(batch_size)], dim = 0)
                noise = torch.randn_like(x4_x_t)
                noise[virtual_node_mask_x4] = 0.0
                if do_partial_inpainting:
                    x4_x_t_inpaint = x4_x_t_inpaint.reshape(batch_size, -1, num_pharm_types)
                    noise = noise.reshape(batch_size, -1, num_pharm_types)[:, :x4_x_t_inpaint.shape[1]]

                    x4_x_t_inpaint = x4_x_t_inpaint + add_noise_to_inpainted_x4_type * noise
                    x4_x_t = x4_x_t.reshape(batch_size, -1, num_pharm_types)
                    x4_x_t[:, :x4_x_t_inpaint.shape[1]] = x4_x_t_inpaint
                    x4_x_t = x4_x_t.reshape(-1, num_pharm_types)
                else:
                    x4_x_t_inpaint = x4_x_t_inpaint + add_noise_to_inpainted_x4_type * noise
                    x4_x_t = x4_x_t_inpaint
        
        
        # get noise parameters for current timestep t and previous timestep prev_t
        noise_params_current = _get_noise_params_for_timestep(params, current_t)
        noise_params_prev = _get_noise_params_for_timestep(params, prev_t) # Need params for interval end
        
        # pass only current params to model input preparation
        x1_params_current = noise_params_current['x1']
        x2_params_current = noise_params_current['x2']
        x3_params_current = noise_params_current['x3']
        x4_params_current = noise_params_current['x4']
        
        # get current data
        input_dict = _prepare_model_input(
            model_pl.model.device, torch.float32, batch_size, current_t,
            x1_pos_t, x1_x_t, x1_batch, x1_bond_edge_x_t, bond_edge_index_x1, virtual_node_mask_x1, x1_params_current,
            x2_pos_t, x2_x_t, x2_batch, virtual_node_mask_x2, x2_params_current,
            x3_pos_t, x3_x_t, x3_batch, virtual_node_mask_x3, x3_params_current,
            x4_pos_t, x4_direction_t, x4_x_t, x4_batch, virtual_node_mask_x4, x4_params_current
        )
        

        # predict noise with neural network    
        with torch.no_grad():
            _, output_dict = model_pl.model.forward(input_dict)
        
        x1_x_out = output_dict['x1']['decoder']['denoiser']['x_out'].detach().cpu()
        x1_bond_edge_x_out = output_dict['x1']['decoder']['denoiser']['bond_edge_x_out'].detach().cpu()
        x1_pos_out = output_dict['x1']['decoder']['denoiser']['pos_out'].detach().cpu()
        x1_pos_out = x1_pos_out - torch_scatter.scatter_mean(x1_pos_out[~virtual_node_mask_x1], x1_batch[~virtual_node_mask_x1], dim = 0)[x1_batch] # removing COM from predicted noise 
        
        x1_x_out[virtual_node_mask_x1, :] = 0.0
        x1_pos_out[virtual_node_mask_x1, :] = 0.0
        
        
        x2_pos_out = output_dict['x2']['decoder']['denoiser']['pos_out']
        if x2_pos_out is not None:
            x2_pos_out = x2_pos_out.detach().cpu() # NOT removing COM from predicted positional noise for x3
            x2_pos_out[virtual_node_mask_x2, :] = 0.0
        else:
            x2_pos_out = torch.zeros_like(x2_pos_t)
            
        
        x3_pos_out = output_dict['x3']['decoder']['denoiser']['pos_out']
        x3_x_out = output_dict['x3']['decoder']['denoiser']['x_out']
        if x3_pos_out is not None:
            x3_pos_out = x3_pos_out.detach().cpu() # NOT removing COM from predicted positional noise for x3
            x3_pos_out[virtual_node_mask_x3, :] = 0.0
            
            x3_x_out = x3_x_out.detach().cpu()
            x3_x_out = x3_x_out.squeeze()
            x3_x_out[virtual_node_mask_x3] = 0.0
        else:
            x3_pos_out = torch.zeros_like(x3_pos_t)
            x3_x_out = torch.zeros_like(x3_x_t)
        
        
        x4_x_out = output_dict['x4']['decoder']['denoiser']['x_out']
        x4_pos_out = output_dict['x4']['decoder']['denoiser']['pos_out']
        x4_direction_out = output_dict['x4']['decoder']['denoiser']['direction_out']
        if x4_x_out is not None:
            x4_pos_out = x4_pos_out.detach().cpu() # NOT removing COM from predicted positional noise for x4
            x4_pos_out[virtual_node_mask_x4, :] = 0.0
            
            x4_direction_out = x4_direction_out.detach().cpu() # NOT removing COM from predicted positional noise for x4
            x4_direction_out[virtual_node_mask_x4, :] = 0.0
            
            x4_x_out = x4_x_out.detach().cpu()
            x4_x_out = x4_x_out.squeeze()
            x4_x_out[virtual_node_mask_x4] = 0.0
        
        else:
            x4_pos_out = torch.zeros_like(x4_pos_t)
            x4_direction_out = torch.zeros_like(x4_direction_t)
            x4_x_out = torch.zeros_like(x4_x_t)
        
        
        
        # Perform reverse denoising step using helper function
        next_state = _perform_reverse_denoising_step(
            current_t, # Pass current time t (tau_i)
            prev_t,    # Pass previous time t-1 (tau_{i-1})
            batch_size, 
            noise_params_current, # Pass params for current time t
            noise_params_prev,   # Pass params for previous time t-1
            sampler_type,        # Pass sampler type
            ddim_eta,            # Pass eta
            # Current states (x_t)
            x1_pos_t, x1_x_t, x1_bond_edge_x_t, x1_batch, virtual_node_mask_x1, 
            x2_pos_t, x2_x_t, x2_batch, virtual_node_mask_x2, 
            x3_pos_t, x3_x_t, x3_batch, virtual_node_mask_x3, 
            x4_pos_t, x4_direction_t, x4_x_t, x4_batch, virtual_node_mask_x4, 
            # Model outputs (predicted noise or x0)
            x1_pos_out, x1_x_out, x1_bond_edge_x_out,
            x2_pos_out, 
            x3_pos_out, x3_x_out,
            x4_pos_out, x4_direction_out, x4_x_out,
            # DDPM specific params (used conditionally within the step func)
            denoising_noise_scale, inject_noise_at_ts, inject_noise_scales
        )

        # Unpack next state
        x1_pos_t_1 = next_state['x1_pos_t_1']
        x1_x_t_1 = next_state['x1_x_t_1']
        x1_bond_edge_x_t_1 = next_state['x1_bond_edge_x_t_1']
        x2_pos_t_1 = next_state['x2_pos_t_1']
        x2_x_t_1 = next_state['x2_x_t_1']
        x3_pos_t_1 = next_state['x3_pos_t_1']
        x3_x_t_1 = next_state['x3_x_t_1']
        x4_pos_t_1 = next_state['x4_pos_t_1']
        x4_direction_t_1 = next_state['x4_direction_t_1']
        x4_x_t_1 = next_state['x4_x_t_1']
        
        
        # saving intermediate states for visualization / tracking
        x1_t_x_list.append(x1_x_t.detach().cpu().numpy())
        x1_t_bond_edge_x_list.append(x1_bond_edge_x_t.detach().cpu().numpy())
        x1_t_pos_list.append(x1_pos_t.detach().cpu().numpy())
        
        x2_t_pos_list.append(x2_pos_t.detach().cpu().numpy())
            
        x3_t_pos_list.append(x3_pos_t.detach().cpu().numpy())
        x3_t_x_list.append(x3_x_t.detach().cpu().numpy())
        
        x4_t_pos_list.append(x4_pos_t.detach().cpu().numpy())
        x4_t_direction_list.append(x4_direction_t.detach().cpu().numpy())
        x4_t_x_list.append(x4_x_t.detach().cpu().numpy())
        
        
        # set next state and iterate
        x1_pos_t = x1_pos_t_1
        x1_x_t = x1_x_t_1
        x1_bond_edge_x_t = x1_bond_edge_x_t_1
        
        x2_pos_t = x2_pos_t_1
        x2_x_t = x2_x_t_1
        
        x3_pos_t = x3_pos_t_1
        x3_x_t = x3_x_t_1
        
        x4_pos_t = x4_pos_t_1
        x4_direction_t = x4_direction_t_1
        x4_x_t = x4_x_t_1
        
        current_time_idx += 1 # Move to next index in time_steps sequence
    
        pbar.update(1)
        
        # this is necessary for clearing CUDA memory
        del output_dict
        del input_dict    
        
    pbar.close()
    
    
    
    ####### Extracting final structures, and re-scaling ########
    
    x2_pos_final = x2_pos_t[~virtual_node_mask_x2].numpy()

    x3_pos_final = x3_pos_t[~virtual_node_mask_x3].numpy()
    x3_x_final = x3_x_t[~virtual_node_mask_x3].numpy()
    x3_x_final = x3_x_final / params['dataset']['x3']['scale_node_features']
    
    x4_x_final = np.argmin(np.abs(x4_x_t[~virtual_node_mask_x4] - params['dataset']['x4']['scale_node_features']), axis = -1)
    x4_x_final = x4_x_final - 1 # readjusting for the previous addition of the virtual node pharmacophore type
    x4_pos_final = x4_pos_t[~virtual_node_mask_x4].numpy()
    
    x4_direction_final = x4_direction_t[~virtual_node_mask_x4].numpy() / params['dataset']['x4']['scale_vector_features']
    x4_direction_final_norm = np.linalg.norm(x4_direction_final, axis = 1)
    x4_direction_final[x4_direction_final_norm < 0.5] = 0.0
    x4_direction_final[x4_direction_final_norm >= 0.5] = x4_direction_final[x4_direction_final_norm >= 0.5] / x4_direction_final_norm[x4_direction_final_norm >= 0.5][..., None]
    
    
    x1_x_t[~virtual_node_mask_x1, 0] = -np.inf # this masks out remaining probability assigned to virtual nodes
    x1_pos_final = x1_pos_t[~virtual_node_mask_x1].numpy()
    x1_x_final = np.argmin(np.abs(x1_x_t[~virtual_node_mask_x1, 0:-len(params['dataset']['x1']['charge_types'])] - params['dataset']['x1']['scale_atom_features']), axis = -1)
    x1_bond_edge_x_final = np.argmin(np.abs(x1_bond_edge_x_t - params['dataset']['x1']['scale_bond_features']), axis = -1)
    
    # need to remap the indices in x1_x_final to the list of atom types
    atomic_number_remapping = torch.tensor([0,1,6,7,8,9,17,35,53,16,15,14]) # [None, 'H', 'C', 'N', 'O', 'F', 'Cl', 'Br', 'I', 'S', 'P', 'Si']
    x1_x_final = atomic_number_remapping[x1_x_final]
    
    
    # return generated structures
    generated_structures = []
    for b in range(batch_size):
        generated_dict = {
            'x1': {
                'atoms': np.split(x1_x_final.numpy(), batch_size)[b],
                #'formal_charges': None, # still need to extract from x1_x_t[~virtual_node_mask_x1, -len(params['dataset']['x1']['charge_types']):]
                'bonds': np.split(x1_bond_edge_x_final.numpy(), batch_size)[b],
                'positions': np.split(x1_pos_final, batch_size)[b],
            },
            'x2': {
                'positions': np.split(x2_pos_final, batch_size)[b],
            },
            'x3': {
                'charges': np.split(x3_x_final, batch_size)[b], # electrostatic potential
                'positions': np.split(x3_pos_final, batch_size)[b],
            },
            'x4': {
                'types': np.split(x4_x_final.numpy(), batch_size)[b],
                'positions': np.split(x4_pos_final, batch_size)[b],
                'directions': np.split(x4_direction_final, batch_size)[b],
            },
        }
        generated_structures.append(generated_dict)
    
    return generated_structures