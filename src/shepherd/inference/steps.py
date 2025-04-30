import torch
import numpy as np
import torch_scatter


# helper function to perform one step of reverse denoising
def _perform_reverse_denoising_step(
    current_t, # current timestep
    prev_t,    # previous timestep
    batch_size, \
    noise_params_current, # dict of params for current_t
    noise_params_prev,    # dict of params for prev_t
    sampler_type,         # 'ddpm' or 'ddim'
    ddim_eta,             # DDIM eta parameter (0=deterministic)
    # current states (x_t)
    x1_pos_t, x1_x_t, x1_bond_edge_x_t, x1_batch, virtual_node_mask_x1, \
    x2_pos_t, x2_x_t, x2_batch, virtual_node_mask_x2, \
    x3_pos_t, x3_x_t, x3_batch, virtual_node_mask_x3, \
    x4_pos_t, x4_direction_t, x4_x_t, x4_batch, virtual_node_mask_x4, \
    # model outputs (predicted noise eps_theta(x_t, t))
    x1_pos_out, x1_x_out, x1_bond_edge_x_out,\
    x2_pos_out, \
    x3_pos_out, x3_x_out,\
    x4_pos_out, x4_direction_out, x4_x_out,\
    # DDPM specific params
    denoising_noise_scale, inject_noise_at_ts, inject_noise_scales):

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
    x1_pos_epsilon = torch.randn(x1_batch_size_nodes, 3)
    x1_pos_epsilon = x1_pos_epsilon - torch_scatter.scatter_mean(x1_pos_epsilon[~virtual_node_mask_x1], x1_batch[~virtual_node_mask_x1], dim=0)[x1_batch]
    x1_pos_epsilon[virtual_node_mask_x1, :] = 0.0
    x1_x_epsilon = torch.randn(x1_batch_size_nodes, num_atom_types)
    x1_x_epsilon[virtual_node_mask_x1, :] = 0.0
    x1_bond_edge_x_epsilon = torch.randn_like(x1_bond_edge_x_out)
    x1_c_t = (noise_params_current['x1']['sigma_t'] * noise_params_current['x1']['sigma_dash_t_1']) / (noise_params_current['x1']['sigma_dash_t'] + 1e-9) if noise_params_current['x1']['t_idx'] > 0 else 0
    x1_c_t = x1_c_t * denoising_noise_scale

    # get added noise - x2
    x2_pos_epsilon = torch.randn(x2_batch_size_nodes, 3)
    x2_pos_epsilon[virtual_node_mask_x2, :] = 0.0
    x2_c_t = (noise_params_current['x2']['sigma_t'] * noise_params_current['x2']['sigma_dash_t_1']) / (noise_params_current['x2']['sigma_dash_t'] + 1e-9) if noise_params_current['x2']['t_idx'] > 0 else 0
    x2_c_t = x2_c_t * denoising_noise_scale

    # get added noise - x3
    x3_pos_epsilon = torch.randn(x3_batch_size_nodes, 3)
    x3_pos_epsilon[virtual_node_mask_x3, :] = 0.0
    x3_x_epsilon = torch.randn(x3_batch_size_nodes)
    x3_x_epsilon[virtual_node_mask_x3, ...] = 0.0
    x3_c_t = (noise_params_current['x3']['sigma_t'] * noise_params_current['x3']['sigma_dash_t_1']) / (noise_params_current['x3']['sigma_dash_t'] + 1e-9) if noise_params_current['x3']['t_idx'] > 0 else 0
    x3_c_t = x3_c_t * denoising_noise_scale

    # get added noise - x4
    x4_pos_epsilon = torch.randn(x4_batch_size_nodes, 3)
    x4_pos_epsilon[virtual_node_mask_x4, :] = 0.0
    x4_direction_epsilon = torch.randn(x4_batch_size_nodes, 3)
    x4_direction_epsilon[virtual_node_mask_x4, :] = 0.0
    x4_x_epsilon = torch.randn(x4_batch_size_nodes, num_pharm_types)
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

    return {
        'x1_pos_t_1': x1_pos_t_1, 'x1_x_t_1': x1_x_t_1, 'x1_bond_edge_x_t_1': x1_bond_edge_x_t_1,
        'x2_pos_t_1': x2_pos_t_1, 'x2_x_t_1': x2_x_t_1, 
        'x3_pos_t_1': x3_pos_t_1, 'x3_x_t_1': x3_x_t_1,
        'x4_pos_t_1': x4_pos_t_1, 'x4_direction_t_1': x4_direction_t_1, 'x4_x_t_1': x4_x_t_1,
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

