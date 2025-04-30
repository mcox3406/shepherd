import torch
import numpy as np
import torch_scatter


# helper function to perform one step of reverse denoising
def _perform_reverse_denoising_step(t, batch_size, noise_params, 
                                    x1_pos_t, x1_x_t, x1_bond_edge_x_t, x1_batch, virtual_node_mask_x1, x1_pos_out, x1_x_out, x1_bond_edge_x_out,
                                    x2_pos_t, x2_x_t, x2_batch, virtual_node_mask_x2, x2_pos_out,
                                    x3_pos_t, x3_x_t, x3_batch, virtual_node_mask_x3, x3_pos_out, x3_x_out,
                                    x4_pos_t, x4_direction_t, x4_x_t, x4_batch, virtual_node_mask_x4, x4_pos_out, x4_direction_out, x4_x_out,
                                    denoising_noise_scale, inject_noise_at_ts, inject_noise_scales):

    x1_params = noise_params['x1']
    x2_params = noise_params['x2']
    x3_params = noise_params['x3']
    x4_params = noise_params['x4']

    # get parameters for current and next timesteps
    x1_t_idx = x1_params['t_idx']
    x1_alpha_t = x1_params['alpha_t']
    x1_sigma_t = x1_params['sigma_t']
    x1_var_dash_t = x1_params['var_dash_t']
    x1_sigma_dash_t = x1_params['sigma_dash_t']
    x1_sigma_dash_t_1 = x1_params['sigma_dash_t_1']

    x2_t_idx = x2_params['t_idx']
    x2_alpha_t = x2_params['alpha_t']
    x2_sigma_t = x2_params['sigma_t']
    x2_var_dash_t = x2_params['var_dash_t']
    x2_sigma_dash_t = x2_params['sigma_dash_t']
    x2_sigma_dash_t_1 = x2_params['sigma_dash_t_1']

    x3_t_idx = x3_params['t_idx']
    x3_alpha_t = x3_params['alpha_t']
    x3_sigma_t = x3_params['sigma_t']
    x3_var_dash_t = x3_params['var_dash_t']
    x3_sigma_dash_t = x3_params['sigma_dash_t']
    x3_sigma_dash_t_1 = x3_params['sigma_dash_t_1']

    x4_t_idx = x4_params['t_idx']
    x4_alpha_t = x4_params['alpha_t']
    x4_sigma_t = x4_params['sigma_t']
    x4_var_dash_t = x4_params['var_dash_t']
    x4_sigma_dash_t = x4_params['sigma_dash_t']
    x4_sigma_dash_t_1 = x4_params['sigma_dash_t_1']
    
    num_atom_types = x1_x_t.shape[-1]
    num_pharm_types = x4_x_t.shape[-1]
    x1_batch_size_nodes = x1_pos_t.shape[0]
    x2_batch_size_nodes = x2_pos_t.shape[0]
    x3_batch_size_nodes = x3_pos_t.shape[0]
    x4_batch_size_nodes = x4_pos_t.shape[0]

    # get added noise - x1
    x1_pos_epsilon = torch.randn(x1_batch_size_nodes, 3)
    x1_pos_epsilon = x1_pos_epsilon - torch_scatter.scatter_mean(x1_pos_epsilon[~virtual_node_mask_x1], x1_batch[~virtual_node_mask_x1], dim=0)[x1_batch]
    x1_pos_epsilon[virtual_node_mask_x1, :] = 0.0
    x1_x_epsilon = torch.randn(x1_batch_size_nodes, num_atom_types)
    x1_x_epsilon[virtual_node_mask_x1, :] = 0.0
    x1_bond_edge_x_epsilon = torch.randn_like(x1_bond_edge_x_out)
    x1_c_t = (x1_sigma_t * x1_sigma_dash_t_1) / (x1_sigma_dash_t + 1e-9) if x1_t_idx > 0 else 0
    x1_c_t = x1_c_t * denoising_noise_scale

    # get added noise - x2
    x2_pos_epsilon = torch.randn(x2_batch_size_nodes, 3)
    x2_pos_epsilon[virtual_node_mask_x2, :] = 0.0
    x2_c_t = (x2_sigma_t * x2_sigma_dash_t_1) / (x2_sigma_dash_t + 1e-9) if x2_t_idx > 0 else 0
    x2_c_t = x2_c_t * denoising_noise_scale

    # get added noise - x3
    x3_pos_epsilon = torch.randn(x3_batch_size_nodes, 3)
    x3_pos_epsilon[virtual_node_mask_x3, :] = 0.0
    x3_x_epsilon = torch.randn(x3_batch_size_nodes)
    x3_x_epsilon[virtual_node_mask_x3, ...] = 0.0
    x3_c_t = (x3_sigma_t * x3_sigma_dash_t_1) / (x3_sigma_dash_t + 1e-9) if x3_t_idx > 0 else 0
    x3_c_t = x3_c_t * denoising_noise_scale

    # get added noise - x4
    x4_pos_epsilon = torch.randn(x4_batch_size_nodes, 3)
    x4_pos_epsilon[virtual_node_mask_x4, :] = 0.0
    x4_direction_epsilon = torch.randn(x4_batch_size_nodes, 3)
    x4_direction_epsilon[virtual_node_mask_x4, :] = 0.0
    x4_x_epsilon = torch.randn(x4_batch_size_nodes, num_pharm_types)
    x4_x_epsilon[virtual_node_mask_x4, ...] = 0.0
    x4_c_t = (x4_sigma_t * x4_sigma_dash_t_1) / (x4_sigma_dash_t + 1e-9) if x4_t_idx > 0 else 0
    x4_c_t = x4_c_t * denoising_noise_scale

    # (intended for symmetry breaking, but could also be used for increasing diversity of samples)
    x1_c_t_injected = x1_c_t
    x2_c_t_injected = x2_c_t
    x3_c_t_injected = x3_c_t
    x4_c_t_injected = x4_c_t
    if len(inject_noise_at_ts) > 0:
        # use a copy to avoid modifying the original list during iteration if called multiple times
        current_inject_ts = list(inject_noise_at_ts)
        current_inject_scales = list(inject_noise_scales)
        if t in current_inject_ts:
            idx_to_pop = current_inject_ts.index(t)
            inject_noise_scale = current_inject_scales[idx_to_pop]
            # Remove from the original lists for next iteration if needed (though typically called once per t)
            # inject_noise_at_ts.pop(idx_to_pop)
            # inject_noise_scales.pop(idx_to_pop)
            
            # extra noisy, only applied to positions to break symmetry
            x1_c_t_injected = x1_c_t + inject_noise_scale
            x2_c_t_injected = x2_c_t + inject_noise_scale
            x3_c_t_injected = x3_c_t + inject_noise_scale
            x4_c_t_injected = x4_c_t + inject_noise_scale

    # reverse denoising step - x1
    x1_pos_t_1 = ((1. / x1_alpha_t) * x1_pos_t) - ((x1_var_dash_t / (x1_alpha_t * x1_sigma_dash_t + 1e-9)) * x1_pos_out) + (x1_c_t_injected * x1_pos_epsilon)
    x1_x_t_1 = ((1. / x1_alpha_t) * x1_x_t) - ((x1_var_dash_t / (x1_alpha_t * x1_sigma_dash_t + 1e-9)) * x1_x_out) + (x1_c_t * x1_x_epsilon)
    x1_bond_edge_x_t_1 = ((1. / x1_alpha_t) * x1_bond_edge_x_t) - ((x1_var_dash_t / (x1_alpha_t * x1_sigma_dash_t + 1e-9)) * x1_bond_edge_x_out) + (x1_c_t * x1_bond_edge_x_epsilon)

    # reverse denoising step - x2
    x2_pos_t_1 = ((1. / float(x2_alpha_t)) * x2_pos_t) - ((x2_var_dash_t / (x2_alpha_t * x2_sigma_dash_t + 1e-9)) * x2_pos_out) + (x2_c_t_injected * x2_pos_epsilon)
    x2_x_t_1 = x2_x_t

    # reverse denoising step - x3
    x3_pos_t_1 = ((1. / float(x3_alpha_t)) * x3_pos_t) - ((x3_var_dash_t / (x3_alpha_t * x3_sigma_dash_t + 1e-9)) * x3_pos_out) + (x3_c_t_injected * x3_pos_epsilon)
    x3_x_t_1 = ((1. / x3_alpha_t) * x3_x_t) - ((x3_var_dash_t / (x3_alpha_t * x3_sigma_dash_t + 1e-9)) * x3_x_out) + (x3_c_t * x3_x_epsilon)

    # reverse denoising step - x4
    x4_pos_t_1 = ((1. / float(x4_alpha_t)) * x4_pos_t) - ((x4_var_dash_t / (x4_alpha_t * x4_sigma_dash_t + 1e-9)) * x4_pos_out) + (x4_c_t_injected * x4_pos_epsilon)
    x4_direction_t_1 = ((1. / float(x4_alpha_t)) * x4_direction_t) - ((x4_var_dash_t / (x4_alpha_t * x4_sigma_dash_t + 1e-9)) * x4_direction_out) + (x4_c_t * x4_direction_epsilon)
    x4_x_t_1 = ((1. / x4_alpha_t) * x4_x_t) - ((x4_var_dash_t / (x4_alpha_t * x4_sigma_dash_t + 1e-9)) * x4_x_out) + (x4_c_t * x4_x_epsilon)

    # resetting virtual nodes (important: do this *after* calculating t-1 state)
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

