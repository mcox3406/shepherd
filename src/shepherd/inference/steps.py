"""
Contains the process to perform one step of reverse denoising.
"""
from typing import Optional
from tqdm import tqdm
import torch
import numpy as np
import torch_scatter

from .noise import (
    _get_noise_params_for_timestep,
    forward_jump
)

# helper function to perform one step of reverse denoising
def _perform_reverse_denoising_step(
    current_t, # current timestep
    prev_t,    # previous timestep
    batch_size,
    noise_params_current, # dict of params for current_t
    noise_params_prev,    # dict of params for prev_t
    sampler_type,         # 'ddpm' or 'ddim'
    ddim_eta,             # DDIM eta parameter (0=deterministic)
    # current states (x_t)
    x1_pos_t, x1_x_t, x1_bond_edge_x_t, x1_batch, virtual_node_mask_x1,
    x2_pos_t, x2_x_t, x2_batch, virtual_node_mask_x2,
    x3_pos_t, x3_x_t, x3_batch, virtual_node_mask_x3,
    x4_pos_t, x4_direction_t, x4_x_t, x4_batch, virtual_node_mask_x4,
    # model outputs (predicted noise eps_theta(x_t, t))
    x1_pos_out, x1_x_out, x1_bond_edge_x_out,
    x2_pos_out,
    x3_pos_out, x3_x_out,
    x4_pos_out, x4_direction_out, x4_x_out,
    # DDPM specific params
    denoising_noise_scale, inject_noise_at_ts, inject_noise_scales,
    noise_dict: Optional[dict] = None):

    # extract parameters for convenience
    # current time
    x1_alpha_dash_t = noise_params_current['x1']['alpha_dash_t']
    x1_var_dash_t = noise_params_current['x1']['var_dash_t']
    x2_alpha_dash_t = noise_params_current['x2']['alpha_dash_t']
    x2_var_dash_t = noise_params_current['x2']['var_dash_t']
    x3_alpha_dash_t = noise_params_current['x3']['alpha_dash_t']
    x3_var_dash_t = noise_params_current['x3']['var_dash_t']
    x4_alpha_dash_t = noise_params_current['x4']['alpha_dash_t']
    x4_var_dash_t = noise_params_current['x4']['var_dash_t']
    
    # previous time
    # Note: DDPM only needs sigma_dash_t_1 from current params for its c_t calc
    x1_alpha_dash_t_prev = noise_params_prev['x1']['alpha_dash_t']
    x1_var_dash_t_prev = noise_params_prev['x1']['var_dash_t'] # = 1 - alpha_dash_t_prev
    x2_alpha_dash_t_prev = noise_params_prev['x2']['alpha_dash_t']
    x2_var_dash_t_prev = noise_params_prev['x2']['var_dash_t']
    x3_alpha_dash_t_prev = noise_params_prev['x3']['alpha_dash_t']
    x3_var_dash_t_prev = noise_params_prev['x3']['var_dash_t']
    x4_alpha_dash_t_prev = noise_params_prev['x4']['alpha_dash_t']
    x4_var_dash_t_prev = noise_params_prev['x4']['var_dash_t']
    
    # get batch sizes and feature dims
    num_atom_types = x1_x_t.shape[-1]
    num_pharm_types = x4_x_t.shape[-1]
    x1_batch_size_nodes = x1_pos_t.shape[0]
    x2_batch_size_nodes = x2_pos_t.shape[0]
    x3_batch_size_nodes = x3_pos_t.shape[0]
    x4_batch_size_nodes = x4_pos_t.shape[0]

    # generate noise epsilon (used by DDPM and DDIM if eta > 0)
    # get added noise - x1
    if noise_dict is not None and 'x1' in noise_dict:
        x1_pos_epsilon = noise_dict['x1']['pos_epsilon'] if 'pos_epsilon' in noise_dict['x1'] else torch.randn(x1_batch_size_nodes, 3)
        x1_x_epsilon = noise_dict['x1']['x_epsilon'] if 'x_epsilon' in noise_dict['x1'] else torch.randn(x1_batch_size_nodes, num_atom_types)
        x1_bond_edge_x_epsilon = noise_dict['x1']['bond_edge_x_epsilon'] if 'bond_edge_x_epsilon' in noise_dict['x1'] else torch.randn_like(x1_bond_edge_x_out)
    else:
        x1_pos_epsilon = torch.randn(x1_batch_size_nodes, 3)
        x1_x_epsilon = torch.randn(x1_batch_size_nodes, num_atom_types)
        x1_bond_edge_x_epsilon = torch.randn_like(x1_bond_edge_x_out)
    x1_pos_epsilon = x1_pos_epsilon - torch_scatter.scatter_mean(x1_pos_epsilon[~virtual_node_mask_x1], x1_batch[~virtual_node_mask_x1], dim=0)[x1_batch]
    x1_pos_epsilon[virtual_node_mask_x1, :] = 0.0
    x1_x_epsilon[virtual_node_mask_x1, :] = 0.0
    x1_c_t = (noise_params_current['x1']['sigma_t'] * noise_params_current['x1']['sigma_dash_t_1']) / (noise_params_current['x1']['sigma_dash_t'] + 1e-9) if noise_params_current['x1']['t_idx'] > 0 else 0
    x1_c_t = x1_c_t * denoising_noise_scale

    # get added noise - x2
    if noise_dict is not None and 'x2' in noise_dict:
        x2_pos_epsilon = noise_dict['x2']['pos_epsilon'] if 'pos_epsilon' in noise_dict['x2'] else torch.randn(x2_batch_size_nodes, 3)
    else:
        x2_pos_epsilon = torch.randn(x2_batch_size_nodes, 3)
    x2_pos_epsilon[virtual_node_mask_x2, :] = 0.0
    x2_c_t = (noise_params_current['x2']['sigma_t'] * noise_params_current['x2']['sigma_dash_t_1']) / (noise_params_current['x2']['sigma_dash_t'] + 1e-9) if noise_params_current['x2']['t_idx'] > 0 else 0
    x2_c_t = x2_c_t * denoising_noise_scale

    # get added noise - x3
    if noise_dict is not None and 'x3' in noise_dict:
        x3_pos_epsilon = noise_dict['x3']['pos_epsilon'] if 'pos_epsilon' in noise_dict['x3'] else torch.randn(x3_batch_size_nodes, 3)
        x3_x_epsilon = noise_dict['x3']['x_epsilon'] if 'x_epsilon' in noise_dict['x3'] else torch.randn(x3_batch_size_nodes)
    else:
        x3_pos_epsilon = torch.randn(x3_batch_size_nodes, 3)
        x3_x_epsilon = torch.randn(x3_batch_size_nodes)
    x3_pos_epsilon[virtual_node_mask_x3, :] = 0.0
    x3_x_epsilon[virtual_node_mask_x3, ...] = 0.0
    x3_c_t = (noise_params_current['x3']['sigma_t'] * noise_params_current['x3']['sigma_dash_t_1']) / (noise_params_current['x3']['sigma_dash_t'] + 1e-9) if noise_params_current['x3']['t_idx'] > 0 else 0
    x3_c_t = x3_c_t * denoising_noise_scale

    # get added noise - x4
    if noise_dict is not None and 'x4' in noise_dict:
        x4_pos_epsilon = noise_dict['x4']['pos_epsilon'] if 'pos_epsilon' in noise_dict['x4'] else torch.randn(x4_batch_size_nodes, 3)
        x4_direction_epsilon = noise_dict['x4']['direction_epsilon'] if 'direction_epsilon' in noise_dict['x4'] else torch.randn(x4_batch_size_nodes, 3)
        x4_x_epsilon = noise_dict['x4']['x_epsilon'] if 'x_epsilon' in noise_dict['x4'] else torch.randn(x4_batch_size_nodes, num_pharm_types)
    else:
        x4_pos_epsilon = torch.randn(x4_batch_size_nodes, 3)
        x4_direction_epsilon = torch.randn(x4_batch_size_nodes, 3)
        x4_x_epsilon = torch.randn(x4_batch_size_nodes, num_pharm_types)
    x4_pos_epsilon[virtual_node_mask_x4, :] = 0.0
    x4_direction_epsilon[virtual_node_mask_x4, :] = 0.0
    x4_x_epsilon[virtual_node_mask_x4, ...] = 0.0
    x4_c_t = (noise_params_current['x4']['sigma_t'] * noise_params_current['x4']['sigma_dash_t_1']) / (noise_params_current['x4']['sigma_dash_t'] + 1e-9) if noise_params_current['x4']['t_idx'] > 0 else 0
    x4_c_t = x4_c_t * denoising_noise_scale

    # --- Conditional Sampler Logic --- 
    if sampler_type == 'ddim':
        # if current_t % 50 == 0 or current_t < 10 or prev_t < 10:
        #      print(f"-- DDIM Step: current_t={current_t}, prev_t={prev_t} --")

        dtype = x1_pos_t.dtype
        device = x1_pos_t.device

        # convert necessary current params to tensors
        x1_alpha_dash_t = torch.tensor(x1_alpha_dash_t, dtype=dtype, device=device)
        x1_var_dash_t = torch.tensor(x1_var_dash_t, dtype=dtype, device=device)
        x2_alpha_dash_t = torch.tensor(x2_alpha_dash_t, dtype=dtype, device=device)
        x2_var_dash_t = torch.tensor(x2_var_dash_t, dtype=dtype, device=device)
        x3_alpha_dash_t = torch.tensor(x3_alpha_dash_t, dtype=dtype, device=device)
        x3_var_dash_t = torch.tensor(x3_var_dash_t, dtype=dtype, device=device)
        x4_alpha_dash_t = torch.tensor(x4_alpha_dash_t, dtype=dtype, device=device)
        x4_var_dash_t = torch.tensor(x4_var_dash_t, dtype=dtype, device=device)

        # convert necessary previous params to tensors
        x1_alpha_dash_t_prev = torch.tensor(x1_alpha_dash_t_prev, dtype=dtype, device=device)
        x1_var_dash_t_prev = torch.tensor(x1_var_dash_t_prev, dtype=dtype, device=device)
        x2_alpha_dash_t_prev = torch.tensor(x2_alpha_dash_t_prev, dtype=dtype, device=device)
        x2_var_dash_t_prev = torch.tensor(x2_var_dash_t_prev, dtype=dtype, device=device)
        x3_alpha_dash_t_prev = torch.tensor(x3_alpha_dash_t_prev, dtype=dtype, device=device)
        x3_var_dash_t_prev = torch.tensor(x3_var_dash_t_prev, dtype=dtype, device=device)
        x4_alpha_dash_t_prev = torch.tensor(x4_alpha_dash_t_prev, dtype=dtype, device=device)
        x4_var_dash_t_prev = torch.tensor(x4_var_dash_t_prev, dtype=dtype, device=device)
        ddim_eta_t = torch.tensor(ddim_eta, dtype=dtype, device=device)

        # Debug print for alphas
        # if current_t % 50 == 0 or current_t < 10 or prev_t < 10:
        #     print(f"  x1_alphas: alpha_t={x1_alpha_dash_t.item():.4f}, alpha_prev={x1_alpha_dash_t_prev.item():.4f}")

        # Calculate predicted x0 (Eq. 9)
        # x0 = (xt - sqrt(1-alpha_t) * eps_theta) / sqrt(alpha_t)
        sqrt_alpha_dash_t_x1 = torch.sqrt(x1_alpha_dash_t)
        sqrt_1m_alpha_dash_t_x1 = torch.sqrt(1.0 - x1_alpha_dash_t)
        pred_x0_pos_x1 = (x1_pos_t - sqrt_1m_alpha_dash_t_x1 * x1_pos_out) / (sqrt_alpha_dash_t_x1 + 1e-9)
        pred_x0_x_x1 = (x1_x_t - sqrt_1m_alpha_dash_t_x1 * x1_x_out) / (sqrt_alpha_dash_t_x1 + 1e-9)
        pred_x0_bond_x1 = (x1_bond_edge_x_t - sqrt_1m_alpha_dash_t_x1 * x1_bond_edge_x_out) / (sqrt_alpha_dash_t_x1 + 1e-9)
        
        sqrt_alpha_dash_t_x2 = torch.sqrt(x2_alpha_dash_t)
        sqrt_1m_alpha_dash_t_x2 = torch.sqrt(1.0 - x2_alpha_dash_t)
        pred_x0_pos_x2 = (x2_pos_t - sqrt_1m_alpha_dash_t_x2 * x2_pos_out) / (sqrt_alpha_dash_t_x2 + 1e-9)
        # x2_x is not diffused

        sqrt_alpha_dash_t_x3 = torch.sqrt(x3_alpha_dash_t)
        sqrt_1m_alpha_dash_t_x3 = torch.sqrt(1.0 - x3_alpha_dash_t)
        pred_x0_pos_x3 = (x3_pos_t - sqrt_1m_alpha_dash_t_x3 * x3_pos_out) / (sqrt_alpha_dash_t_x3 + 1e-9)
        pred_x0_x_x3 = (x3_x_t - sqrt_1m_alpha_dash_t_x3 * x3_x_out) / (sqrt_alpha_dash_t_x3 + 1e-9)

        sqrt_alpha_dash_t_x4 = torch.sqrt(x4_alpha_dash_t)
        sqrt_1m_alpha_dash_t_x4 = torch.sqrt(1.0 - x4_alpha_dash_t)
        pred_x0_pos_x4 = (x4_pos_t - sqrt_1m_alpha_dash_t_x4 * x4_pos_out) / (sqrt_alpha_dash_t_x4 + 1e-9)
        pred_x0_dir_x4 = (x4_direction_t - sqrt_1m_alpha_dash_t_x4 * x4_direction_out) / (sqrt_alpha_dash_t_x4 + 1e-9)
        pred_x0_x_x4 = (x4_x_t - sqrt_1m_alpha_dash_t_x4 * x4_x_out) / (sqrt_alpha_dash_t_x4 + 1e-9)
        
        # Debug print for predicted x0 (summary)
        # if current_t % 50 == 0 or current_t < 10 or prev_t < 10:
        #     print(f"  pred_x0_pos_x1: mean={pred_x0_pos_x1.mean().item():.4f}, std={pred_x0_pos_x1.std().item():.4f}")

        # calculate DDIM sigma (dependent on eta)
        # sigma^2 = eta^2 * (1 - alpha_{t-1}) / (1 - alpha_t) * (1 - alpha_t / alpha_{t-1})
        # clamp arguments to sqrt to avoid NaNs
        def calculate_sigma_sq(var_dash_t, var_dash_t_prev, alpha_dash_t, alpha_dash_t_prev, eta):
             term1 = (var_dash_t_prev / (var_dash_t + 1e-9)) if var_dash_t > 1e-9 else torch.tensor(0.0, device=eta.device, dtype=eta.dtype)
             ratio_alpha = (alpha_dash_t / (alpha_dash_t_prev+1e-9)) if alpha_dash_t_prev > 1e-9 else torch.tensor(0.0, device=eta.device, dtype=eta.dtype)
             term2 = 1.0 - ratio_alpha
             sigma_sq = (eta**2) * term1 * term2
             return torch.clamp(sigma_sq, min=0.0)
        
        sigma_sq_x1 = calculate_sigma_sq(x1_var_dash_t, x1_var_dash_t_prev, x1_alpha_dash_t, x1_alpha_dash_t_prev, ddim_eta_t)
        sigma_sq_x2 = calculate_sigma_sq(x2_var_dash_t, x2_var_dash_t_prev, x2_alpha_dash_t, x2_alpha_dash_t_prev, ddim_eta_t)
        sigma_sq_x3 = calculate_sigma_sq(x3_var_dash_t, x3_var_dash_t_prev, x3_alpha_dash_t, x3_alpha_dash_t_prev, ddim_eta_t)
        sigma_sq_x4 = calculate_sigma_sq(x4_var_dash_t, x4_var_dash_t_prev, x4_alpha_dash_t, x4_alpha_dash_t_prev, ddim_eta_t)
        
        sigma_val_x1 = torch.sqrt(sigma_sq_x1)
        sigma_val_x2 = torch.sqrt(sigma_sq_x2)
        sigma_val_x3 = torch.sqrt(sigma_sq_x3)
        sigma_val_x4 = torch.sqrt(sigma_sq_x4)
        
        # Debug print for sigmas
        # if current_t % 50 == 0 or current_t < 10 or prev_t < 10:
        #     print(f"  sigmas_x1: sigma_sq={sigma_sq_x1.item():.4e}, sigma_val={sigma_val_x1.item():.4e}")

        # calculate coefficient for direction pointing to x_t
        # sqrt(1 - alpha_{t-1} - sigma^2)
        sqrt_1m_alpha_prev_minus_sigma_sq_x1 = torch.sqrt(torch.clamp(1.0 - x1_alpha_dash_t_prev - sigma_sq_x1, min=0.0))
        sqrt_1m_alpha_prev_minus_sigma_sq_x2 = torch.sqrt(torch.clamp(1.0 - x2_alpha_dash_t_prev - sigma_sq_x2, min=0.0))
        sqrt_1m_alpha_prev_minus_sigma_sq_x3 = torch.sqrt(torch.clamp(1.0 - x3_alpha_dash_t_prev - sigma_sq_x3, min=0.0))
        sqrt_1m_alpha_prev_minus_sigma_sq_x4 = torch.sqrt(torch.clamp(1.0 - x4_alpha_dash_t_prev - sigma_sq_x4, min=0.0))
        
        # Debug print for direction coefficient sqrt term
        # if current_t % 50 == 0 or current_t < 10 or prev_t < 10:
        #     print(f"  sqrt_1m_alpha_prev_minus_sigma_sq_x1: {sqrt_1m_alpha_prev_minus_sigma_sq_x1.item():.4f}")
        
        # predicted noise term (eps_theta)
        noise_term_pos_x1 = x1_pos_out # model directly predicts noise
        noise_term_x_x1 = x1_x_out
        noise_term_bond_x1 = x1_bond_edge_x_out
        noise_term_pos_x2 = x2_pos_out
        noise_term_pos_x3 = x3_pos_out
        noise_term_x_x3 = x3_x_out
        noise_term_pos_x4 = x4_pos_out
        noise_term_dir_x4 = x4_direction_out
        noise_term_x_x4 = x4_x_out
        
        # first term coefficient (sqrt(alpha_{t-1}))
        sqrt_alpha_dash_t_prev_x1 = torch.sqrt(x1_alpha_dash_t_prev)
        sqrt_alpha_dash_t_prev_x2 = torch.sqrt(x2_alpha_dash_t_prev)
        sqrt_alpha_dash_t_prev_x3 = torch.sqrt(x3_alpha_dash_t_prev)
        sqrt_alpha_dash_t_prev_x4 = torch.sqrt(x4_alpha_dash_t_prev)

        # combine terms for final x_{t-1} update
        x1_pos_t_1 = sqrt_alpha_dash_t_prev_x1 * pred_x0_pos_x1 + \
                       sqrt_1m_alpha_prev_minus_sigma_sq_x1 * noise_term_pos_x1 + \
                       sigma_val_x1 * x1_pos_epsilon
        x1_x_t_1 = sqrt_alpha_dash_t_prev_x1 * pred_x0_x_x1 + \
                     sqrt_1m_alpha_prev_minus_sigma_sq_x1 * noise_term_x_x1 + \
                     sigma_val_x1 * x1_x_epsilon
        x1_bond_edge_x_t_1 = sqrt_alpha_dash_t_prev_x1 * pred_x0_bond_x1 + \
                             sqrt_1m_alpha_prev_minus_sigma_sq_x1 * noise_term_bond_x1 + \
                             sigma_val_x1 * x1_bond_edge_x_epsilon
        
        x2_pos_t_1 = sqrt_alpha_dash_t_prev_x2 * pred_x0_pos_x2 + \
                       sqrt_1m_alpha_prev_minus_sigma_sq_x2 * noise_term_pos_x2 + \
                       sigma_val_x2 * x2_pos_epsilon
        x2_x_t_1 = x2_x_t # not diffused

        x3_pos_t_1 = sqrt_alpha_dash_t_prev_x3 * pred_x0_pos_x3 + \
                       sqrt_1m_alpha_prev_minus_sigma_sq_x3 * noise_term_pos_x3 + \
                       sigma_val_x3 * x3_pos_epsilon
        x3_x_t_1 = sqrt_alpha_dash_t_prev_x3 * pred_x0_x_x3 + \
                     sqrt_1m_alpha_prev_minus_sigma_sq_x3 * noise_term_x_x3 + \
                     sigma_val_x3 * x3_x_epsilon

        x4_pos_t_1 = sqrt_alpha_dash_t_prev_x4 * pred_x0_pos_x4 + \
                       sqrt_1m_alpha_prev_minus_sigma_sq_x4 * noise_term_pos_x4 + \
                       sigma_val_x4 * x4_pos_epsilon
        x4_direction_t_1 = sqrt_alpha_dash_t_prev_x4 * pred_x0_dir_x4 + \
                             sqrt_1m_alpha_prev_minus_sigma_sq_x4 * noise_term_dir_x4 + \
                             sigma_val_x4 * x4_direction_epsilon
        x4_x_t_1 = sqrt_alpha_dash_t_prev_x4 * pred_x0_x_x4 + \
                     sqrt_1m_alpha_prev_minus_sigma_sq_x4 * noise_term_x_x4 + \
                     sigma_val_x4 * x4_x_epsilon

        # Debug print for final state x1 (summary)
        # if current_t % 50 == 0 or current_t < 10 or prev_t < 10:
        #     print(f"  final_x1_pos_t_1: mean={x1_pos_t_1.mean().item():.4f}, std={x1_pos_t_1.std().item():.4f}")

    elif sampler_type == 'ddpm':
        # fetch necessary single-step and cumulative params for current_t
        x1_t_idx = noise_params_current['x1']['t_idx']
        x1_alpha_t = noise_params_current['x1']['alpha_t'] # single step alpha
        x1_sigma_t = noise_params_current['x1']['sigma_t']
        x1_sigma_dash_t = noise_params_current['x1']['sigma_dash_t']
        x1_sigma_dash_t_1 = noise_params_current['x1']['sigma_dash_t_1'] # from prev step in full schedule
        # Note: DDPM formula uses single-step alpha_t and cumulative vars/sigmas
        # var_dash_t is needed from current params
        x1_var_dash_t = noise_params_current['x1']['var_dash_t'] 
        
        x2_t_idx = noise_params_current['x2']['t_idx']
        x2_alpha_t = noise_params_current['x2']['alpha_t']
        x2_sigma_t = noise_params_current['x2']['sigma_t']
        x2_sigma_dash_t = noise_params_current['x2']['sigma_dash_t']
        x2_sigma_dash_t_1 = noise_params_current['x2']['sigma_dash_t_1']
        x2_var_dash_t = noise_params_current['x2']['var_dash_t']

        x3_t_idx = noise_params_current['x3']['t_idx']
        x3_alpha_t = noise_params_current['x3']['alpha_t']
        x3_sigma_t = noise_params_current['x3']['sigma_t']
        x3_sigma_dash_t = noise_params_current['x3']['sigma_dash_t']
        x3_sigma_dash_t_1 = noise_params_current['x3']['sigma_dash_t_1']
        x3_var_dash_t = noise_params_current['x3']['var_dash_t']

        x4_t_idx = noise_params_current['x4']['t_idx']
        x4_alpha_t = noise_params_current['x4']['alpha_t']
        x4_sigma_t = noise_params_current['x4']['sigma_t']
        x4_sigma_dash_t = noise_params_current['x4']['sigma_dash_t']
        x4_sigma_dash_t_1 = noise_params_current['x4']['sigma_dash_t_1']
        x4_var_dash_t = noise_params_current['x4']['var_dash_t']

        # calculate noise scale factor 'c_t' using current schedule params
        x1_c_t = (x1_sigma_t * x1_sigma_dash_t_1) / (x1_sigma_dash_t + 1e-9) if x1_t_idx > 0 else 0
        x1_c_t = x1_c_t * denoising_noise_scale
        x2_c_t = (x2_sigma_t * x2_sigma_dash_t_1) / (x2_sigma_dash_t + 1e-9) if x2_t_idx > 0 else 0
        x2_c_t = x2_c_t * denoising_noise_scale
        x3_c_t = (x3_sigma_t * x3_sigma_dash_t_1) / (x3_sigma_dash_t + 1e-9) if x3_t_idx > 0 else 0
        x3_c_t = x3_c_t * denoising_noise_scale
        x4_c_t = (x4_sigma_t * x4_sigma_dash_t_1) / (x4_sigma_dash_t + 1e-9) if x4_t_idx > 0 else 0
        x4_c_t = x4_c_t * denoising_noise_scale

        # apply noise injection logic using current_t
        x1_c_t_injected = x1_c_t
        x2_c_t_injected = x2_c_t
        x3_c_t_injected = x3_c_t
        x4_c_t_injected = x4_c_t
        # use copies to avoid modifying lists if function is somehow called in a loop over t
        current_inject_ts = list(inject_noise_at_ts)
        current_inject_scales = list(inject_noise_scales)
        if current_t in current_inject_ts:
            idx_to_pop = current_inject_ts.index(current_t)
            inject_noise_scale = current_inject_scales[idx_to_pop]
            # Don't pop from original lists 
            x1_c_t_injected = x1_c_t + inject_noise_scale
            x2_c_t_injected = x2_c_t + inject_noise_scale
            x3_c_t_injected = x3_c_t + inject_noise_scale
            x4_c_t_injected = x4_c_t + inject_noise_scale

        # apply the original DDPM update rule
        # xt-1 = (xt - (1-alpha_t)/sqrt(1-alpha_dash_t) * eps_theta) / sqrt(alpha_t) + sigma_t * eps
        # original implementation uses: (xt/alpha_t) - (var_dash_t / (alpha_t * sigma_dash_t)) * eps_theta + c_t * eps
        x1_pos_t_1 = ((1. / x1_alpha_t) * x1_pos_t) - ((x1_var_dash_t / (x1_alpha_t * x1_sigma_dash_t + 1e-9)) * x1_pos_out) + (x1_c_t_injected * x1_pos_epsilon)
        x1_x_t_1 = ((1. / x1_alpha_t) * x1_x_t) - ((x1_var_dash_t / (x1_alpha_t * x1_sigma_dash_t + 1e-9)) * x1_x_out) + (x1_c_t * x1_x_epsilon)
        x1_bond_edge_x_t_1 = ((1. / x1_alpha_t) * x1_bond_edge_x_t) - ((x1_var_dash_t / (x1_alpha_t * x1_sigma_dash_t + 1e-9)) * x1_bond_edge_x_out) + (x1_c_t * x1_bond_edge_x_epsilon)

        x2_pos_t_1 = ((1. / float(x2_alpha_t)) * x2_pos_t) - ((x2_var_dash_t / (x2_alpha_t * x2_sigma_dash_t + 1e-9)) * x2_pos_out) + (x2_c_t_injected * x2_pos_epsilon)
        x2_x_t_1 = x2_x_t # Not diffused

        x3_pos_t_1 = ((1. / float(x3_alpha_t)) * x3_pos_t) - ((x3_var_dash_t / (x3_alpha_t * x3_sigma_dash_t + 1e-9)) * x3_pos_out) + (x3_c_t_injected * x3_pos_epsilon)
        x3_x_t_1 = ((1. / x3_alpha_t) * x3_x_t) - ((x3_var_dash_t / (x3_alpha_t * x3_sigma_dash_t + 1e-9)) * x3_x_out) + (x3_c_t * x3_x_epsilon)

        x4_pos_t_1 = ((1. / float(x4_alpha_t)) * x4_pos_t) - ((x4_var_dash_t / (x4_alpha_t * x4_sigma_dash_t + 1e-9)) * x4_pos_out) + (x4_c_t_injected * x4_pos_epsilon)
        x4_direction_t_1 = ((1. / float(x4_alpha_t)) * x4_direction_t) - ((x4_var_dash_t / (x4_alpha_t * x4_sigma_dash_t + 1e-9)) * x4_direction_out) + (x4_c_t * x4_direction_epsilon)
        x4_x_t_1 = ((1. / x4_alpha_t) * x4_x_t) - ((x4_var_dash_t / (x4_alpha_t * x4_sigma_dash_t + 1e-9)) * x4_x_out) + (x4_c_t * x4_x_epsilon)

    else:
        raise ValueError(f"Unsupported sampler_type: {sampler_type}")

    # reset virtual nodes (common to both paths)
    x1_pos_t_1[virtual_node_mask_x1] = x1_pos_t[virtual_node_mask_x1]
    x1_x_t_1[virtual_node_mask_x1] = x1_x_t[virtual_node_mask_x1]
    x2_pos_t_1[virtual_node_mask_x2] = x2_pos_t[virtual_node_mask_x2]
    x2_x_t_1[virtual_node_mask_x2] = x2_x_t[virtual_node_mask_x2]
    x3_pos_t_1[virtual_node_mask_x3] = x3_pos_t[virtual_node_mask_x3]
    x3_x_t_1[virtual_node_mask_x3] = x3_x_t[virtual_node_mask_x3]
    x4_pos_t_1[virtual_node_mask_x4] = x4_pos_t[virtual_node_mask_x4]
    x4_direction_t_1[virtual_node_mask_x4] = x4_direction_t[virtual_node_mask_x4]
    x4_x_t_1[virtual_node_mask_x4] = x4_x_t[virtual_node_mask_x4]

    if noise_dict is not None:
        noise_dict = None

    return {
        'x1_pos_t_1': x1_pos_t_1, 'x1_x_t_1': x1_x_t_1, 'x1_bond_edge_x_t_1': x1_bond_edge_x_t_1,
        'x2_pos_t_1': x2_pos_t_1, 'x2_x_t_1': x2_x_t_1, 
        'x3_pos_t_1': x3_pos_t_1, 'x3_x_t_1': x3_x_t_1,
        'x4_pos_t_1': x4_pos_t_1, 'x4_direction_t_1': x4_direction_t_1, 'x4_x_t_1': x4_x_t_1,
        'noise_dict': noise_dict
    }


# helper function to prepare the model input dictionary
def _prepare_model_input(device, dtype, batch_size, t, 
                         x1_pos_t, x1_x_t, x1_batch, x1_bond_edge_x_t, x1_bond_edge_index, x1_virtual_node_mask, x1_params, 
                         x2_pos_t, x2_x_t, x2_batch, x2_virtual_node_mask, x2_params, 
                         x3_pos_t, x3_x_t, x3_batch, x3_virtual_node_mask, x3_params, 
                         x4_pos_t, x4_direction_t, x4_x_t, x4_batch, x4_virtual_node_mask, x4_params):
    
    x1_timestep = torch.tensor([t] * batch_size)
    x2_timestep = torch.tensor([t] * batch_size)
    x3_timestep = torch.tensor([t] * batch_size)
    x4_timestep = torch.tensor([t] * batch_size)

    x1_sigma_dash_t_ = torch.tensor([x1_params['sigma_dash_t']] * batch_size, dtype=dtype)
    x1_alpha_dash_t_ = torch.tensor([x1_params['alpha_dash_t']] * batch_size, dtype=dtype)
    
    x2_sigma_dash_t_ = torch.tensor([x2_params['sigma_dash_t']] * batch_size, dtype=dtype)
    x2_alpha_dash_t_ = torch.tensor([x2_params['alpha_dash_t']] * batch_size, dtype=dtype)
    
    x3_sigma_dash_t_ = torch.tensor([x3_params['sigma_dash_t']] * batch_size, dtype=dtype)
    x3_alpha_dash_t_ = torch.tensor([x3_params['alpha_dash_t']] * batch_size, dtype=dtype)
    
    x4_sigma_dash_t_ = torch.tensor([x4_params['sigma_dash_t']] * batch_size, dtype=dtype)
    x4_alpha_dash_t_ = torch.tensor([x4_params['alpha_dash_t']] * batch_size, dtype=dtype)

    input_dict = {
        'device': device,
        'dtype': dtype,
        'x1': {
            'decoder': {
                'pos': x1_pos_t.to(device),
                'x': x1_x_t.to(device),
                'batch': x1_batch.to(device),
                'bond_edge_x': x1_bond_edge_x_t.to(device),
                'bond_edge_index': x1_bond_edge_index.to(device),
                'timestep': x1_timestep.to(device),
                'sigma_dash_t': x1_sigma_dash_t_.to(device),
                'alpha_dash_t': x1_alpha_dash_t_.to(device),
                'virtual_node_mask': x1_virtual_node_mask.to(device),
            },
        },
        'x2': {
            'decoder': {
                'pos': x2_pos_t.to(device),
                'x': x2_x_t.to(device),
                'batch': x2_batch.to(device),
                'timestep': x2_timestep.to(device),
                'sigma_dash_t': x2_sigma_dash_t_.to(device),
                'alpha_dash_t': x2_alpha_dash_t_.to(device),
                'virtual_node_mask': x2_virtual_node_mask.to(device),
            },
        },
        'x3': {
            'decoder': {
                'pos': x3_pos_t.to(device),
                'x': x3_x_t.to(device),
                'batch': x3_batch.to(device),
                'timestep': x3_timestep.to(device),
                'sigma_dash_t': x3_sigma_dash_t_.to(device),
                'alpha_dash_t': x3_alpha_dash_t_.to(device),
                'virtual_node_mask': x3_virtual_node_mask.to(device),
            },
        },
        'x4': {
            'decoder': {
                'x': x4_x_t.to(device),
                'pos': x4_pos_t.to(device),
                'direction': x4_direction_t.to(device),
                'batch': x4_batch.to(device),
                'timestep': x4_timestep.to(device),
                'sigma_dash_t': x4_sigma_dash_t_.to(device),
                'alpha_dash_t': x4_alpha_dash_t_.to(device),
                'virtual_node_mask': x4_virtual_node_mask.to(device),
            },
        },
    }
    return input_dict


def _pack_inpainting_dict(
    x2_pos_inpainting_trajectory, x3_pos_inpainting_trajectory, x3_x_inpainting_trajectory,
    x4_pos_inpainting_trajectory, x4_direction_inpainting_trajectory, x4_x_inpainting_trajectory,
    stop_inpainting_at_time_x2, stop_inpainting_at_time_x3, stop_inpainting_at_time_x4,
    add_noise_to_inpainted_x2_pos, add_noise_to_inpainted_x3_pos, add_noise_to_inpainted_x3_x,
    add_noise_to_inpainted_x4_pos, add_noise_to_inpainted_x4_direction, add_noise_to_inpainted_x4_type,
    do_partial_inpainting
):
    return {
        'x2_pos_inpainting_trajectory': x2_pos_inpainting_trajectory,
        'x3_pos_inpainting_trajectory': x3_pos_inpainting_trajectory,
        'x3_x_inpainting_trajectory': x3_x_inpainting_trajectory,
        'x4_pos_inpainting_trajectory': x4_pos_inpainting_trajectory,
        'x4_direction_inpainting_trajectory': x4_direction_inpainting_trajectory,
        'x4_x_inpainting_trajectory': x4_x_inpainting_trajectory,
        'stop_inpainting_at_time_x2': stop_inpainting_at_time_x2,
        'stop_inpainting_at_time_x3': stop_inpainting_at_time_x3,
        'stop_inpainting_at_time_x4': stop_inpainting_at_time_x4,
        'add_noise_to_inpainted_x2_pos': add_noise_to_inpainted_x2_pos,
        'add_noise_to_inpainted_x3_pos': add_noise_to_inpainted_x3_pos,
        'add_noise_to_inpainted_x3_x': add_noise_to_inpainted_x3_x,
        'add_noise_to_inpainted_x4_pos': add_noise_to_inpainted_x4_pos,
        'add_noise_to_inpainted_x4_direction': add_noise_to_inpainted_x4_direction,
        'add_noise_to_inpainted_x4_type': add_noise_to_inpainted_x4_type,
        'do_partial_inpainting': do_partial_inpainting,
    }


def _inference_step(
    model_pl, params,
    # times
    time_steps, current_time_idx,
    harmonize, harmonize_ts, harmonize_jumps,
    batch_size,
    sampler_type, ddim_eta,
    denoising_noise_scale, inject_noise_at_ts, inject_noise_scales,
    # current states
    x1_pos_t, x1_x_t, x1_bond_edge_x_t, x1_batch, bond_edge_index_x1, virtual_node_mask_x1,
    x2_pos_t, x2_x_t, x2_batch, virtual_node_mask_x2,
    x3_pos_t, x3_x_t, x3_batch, virtual_node_mask_x3,
    x4_pos_t, x4_direction_t, x4_x_t, x4_batch, virtual_node_mask_x4,
    # noise
    noise_dict,
    # inpainting
    inpaint_x2_pos, inpaint_x3_pos, inpaint_x3_x, inpaint_x4_pos, inpaint_x4_direction, inpaint_x4_type,
    inpainting_dict: Optional[dict] = None,
    # progress bar
    pbar: Optional[tqdm] = None
    ):
    """
    Inner loop for the denoising process.
    This function is called by the main denoising function and handles the
    denoising steps for each time step in the sequence.
    """
    current_t = time_steps[current_time_idx]
    prev_t = time_steps[current_time_idx + 1] # The time we are calculating state FOR
    
    # t passed to helpers/model should be current time
    t = current_t 

    # inputs (these might be redundant now, t is the main driver)
    x1_t = t
    x2_t = t
    x3_t = t
    x4_t = t

    stop_inpainting_at_time_x2 = 0.0
    stop_inpainting_at_time_x3 = 0.0
    stop_inpainting_at_time_x4 = 0.0

    if inpainting_dict is not None:
        if inpaint_x2_pos:
            x2_pos_inpainting_trajectory = inpainting_dict['x2_pos_inpainting_trajectory']
            stop_inpainting_at_time_x2 = inpainting_dict['stop_inpainting_at_time_x2']
            add_noise_to_inpainted_x2_pos = inpainting_dict['add_noise_to_inpainted_x2_pos']
        else:
            stop_inpainting_at_time_x2 = 0.0
        if inpaint_x3_pos:
            x3_pos_inpainting_trajectory = inpainting_dict['x3_pos_inpainting_trajectory']
        if inpaint_x3_x:
            x3_x_inpainting_trajectory = inpainting_dict['x3_x_inpainting_trajectory']
        if inpaint_x3_pos or inpaint_x3_x:
            stop_inpainting_at_time_x3 = inpainting_dict['stop_inpainting_at_time_x3']
            add_noise_to_inpainted_x3_pos = inpainting_dict['add_noise_to_inpainted_x3_pos']
            add_noise_to_inpainted_x3_x = inpainting_dict['add_noise_to_inpainted_x3_x']
        else:
            stop_inpainting_at_time_x3 = 0.0
        if inpaint_x4_pos:
            x4_pos_inpainting_trajectory = inpainting_dict['x4_pos_inpainting_trajectory']
        if inpaint_x4_direction:
            x4_direction_inpainting_trajectory = inpainting_dict['x4_direction_inpainting_trajectory']
        if inpaint_x4_type:
            x4_x_inpainting_trajectory = inpainting_dict['x4_x_inpainting_trajectory']
        if inpaint_x4_pos or inpaint_x4_direction or inpaint_x4_type:
            stop_inpainting_at_time_x4 = inpainting_dict['stop_inpainting_at_time_x4']
            add_noise_to_inpainted_x4_pos = inpainting_dict['add_noise_to_inpainted_x4_pos']
            add_noise_to_inpainted_x4_direction = inpainting_dict['add_noise_to_inpainted_x4_direction']
            add_noise_to_inpainted_x4_type = inpainting_dict['add_noise_to_inpainted_x4_type']
        else:
            stop_inpainting_at_time_x4 = 0.0
        do_partial_inpainting = inpainting_dict['do_partial_inpainting']
    
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
            if pbar is not None:
                pbar.update(harmonize_jump_len) # update progress bar for the jumped steps
            print(f"Harmonization jumped from t={t} to t={jumped_to_t}, resuming loop.")
            t = jumped_to_t # update t for the next iteration start
            # need to re-fetch noise params for the new 't' before proceeding if the loop continued immediately,
            # but we will recalculate at the start of the next iteration anyway
            next_state = {
                'x1_pos_t_1': x1_pos_t, 'x1_x_t_1': x1_x_t, 'x1_bond_edge_x_t_1': x1_bond_edge_x_t,
                'x2_pos_t_1': x2_pos_t, 'x2_x_t_1': x2_x_t, 
                'x3_pos_t_1': x3_pos_t, 'x3_x_t_1': x3_x_t,
                'x4_pos_t_1': x4_pos_t, 'x4_direction_t_1': x4_direction_t, 'x4_x_t_1': x4_x_t,
                'noise_dict': noise_dict
            }
            return current_time_idx, next_state # skip the rest of the current loop iteration (denoising step)
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
            num_pharm_types = params['dataset']['x4']['max_node_types']
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
        denoising_noise_scale, inject_noise_at_ts, inject_noise_scales,
        noise_dict=noise_dict
    )

    # if save_intermediate:
    #     # saving intermediate states for visualization / tracking
    #     x1_t_x_list.append(x1_x_t.detach().cpu().numpy())
    #     x1_t_bond_edge_x_list.append(x1_bond_edge_x_t.detach().cpu().numpy())
    #     x1_t_pos_list.append(x1_pos_t.detach().cpu().numpy())
        
    #     x2_t_pos_list.append(x2_pos_t.detach().cpu().numpy())
            
    #     x3_t_pos_list.append(x3_pos_t.detach().cpu().numpy())
    #     x3_t_x_list.append(x3_x_t.detach().cpu().numpy())
        
    #     x4_t_pos_list.append(x4_pos_t.detach().cpu().numpy())
    #     x4_t_direction_list.append(x4_direction_t.detach().cpu().numpy())
    #     x4_t_x_list.append(x4_x_t.detach().cpu().numpy())

    if pbar is not None:
        pbar.update(1)

    del output_dict
    del input_dict

    current_time_idx += 1 # Move to next index in time_steps sequence

    return current_time_idx, next_state # next_state is a dictionary with updated states


def _extract_generated_samples(
        x1_x_t, x1_pos_t, x1_bond_edge_x_t, virtual_node_mask_x1,
        x2_pos_t, virtual_node_mask_x2,
        x3_pos_t, x3_x_t, virtual_node_mask_x3,
        x4_pos_t, x4_direction_t, x4_x_t, virtual_node_mask_x4,
        params, batch_size):
    """
    Extracting final structures, and re-scaling
    """
    
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
