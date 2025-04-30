import numpy as np
import torch
import torch_scatter


# harmonization functions
def forward_jump_parameters(t_start_idx, jump, sigma_ts):
    t_end_idx = t_start_idx + jump
    sigma_ts_ = sigma_ts[t_start_idx : t_end_idx] # std deviation schedule
    alpha_ts_ = (1. - sigma_ts_** 2.0)**0.5 # how much (mean) signal is preserved at each noising step
    alpha_dash_ts_ = np.cumprod(alpha_ts_)
    var_dash_ts_ = 1. - alpha_dash_ts_**2.0
    sigma_dash_ts_ = var_dash_ts_**0.5
    return alpha_dash_ts_[-1], sigma_dash_ts_[-1], t_end_idx

def forward_jump(x, t_start, jump, sigma_ts, remove_COM_from_noise = False, batch = None, mask = None):
    assert t_start + jump <= sigma_ts.shape[0] + 1 # can't jump past t=T
    
    if mask is None:
        mask = torch.zeros(x.shape, dtype = torch.long) == 0
    
    t_start_idx = t_start - 1
    alpha_jump, sigma_jump, t_end_idx = forward_jump_parameters(t_start_idx, jump, sigma_ts)
    t_end = t_end_idx + 1
    
    noise = torch.randn(x.shape)
    if remove_COM_from_noise:
        assert batch is not None
        assert len(x.shape) == 2
        # noise must have shape (N, 3)
        assert noise.shape[1] == 3
        noise = noise - torch_scatter.scatter_mean(noise[mask], batch[mask], dim = 0)[batch]
    noise[~mask, ...] = 0.0
    
    x_jump = alpha_jump * x + sigma_jump * noise
    x_jump[~mask] = x[~mask]
    
    return x_jump, t_end

# simulate forward noising trajectory
def forward_trajectory(x, ts, alpha_ts, sigma_ts, remove_COM_from_noise = False, mask = None, deterministic = False):
    if mask is None:
        mask = torch.ones(x.shape[0]) == 1.0
    
    if remove_COM_from_noise:
        assert x.shape[1] == 3
    
    trajectory = {0:x}
    
    x_t = x
    for t_idx, t in enumerate(ts):
        
        alpha_t = alpha_ts[t_idx]
        sigma_t = sigma_ts[t_idx]
        
        noise = torch.randn(x.shape)
        if remove_COM_from_noise:
            noise = noise - torch.mean(noise[mask], dim = 0)
        noise[~mask, ...] = 0.0
        
        if deterministic:
            noise = 0.0
        
        x_t_plus_1 = alpha_t * x_t + sigma_t * noise
        x_t_plus_1[~mask, ...] = x_t[~mask, ...]
        
        trajectory[t] = x_t_plus_1
        
        x_t = x_t_plus_1
    
    return trajectory


# helper function to get noise parameters for a given timestep
def _get_noise_params_for_timestep(params, t):
    noise_params = {}
    modalities = ['x1', 'x2', 'x3', 'x4']

    for mod in modalities:
        schedule = params['noise_schedules'][mod]
        ts_array = schedule['ts']
        t_idx = np.where(ts_array == t)[0]
        
        if len(t_idx) == 0:
            # handle edge case if t is not in the schedule (e.g., after harmonization jump)
            # find the closest timestep in the schedule <= t
            valid_indices = np.where(ts_array <= t)[0]
            if len(valid_indices) == 0:
                raise ValueError(f"Timestep {t} is smaller than the smallest timestep in the schedule for {mod}")
            t_idx = valid_indices[-1]
        else:
            t_idx = t_idx[0]

        mod_params = {
            't_idx': t_idx,
            'alpha_t': schedule['alpha_ts'][t_idx],
            'sigma_t': schedule['sigma_ts'][t_idx],
            'alpha_dash_t': schedule['alpha_dash_ts'][t_idx],
            'var_dash_t': schedule['var_dash_ts'][t_idx],
            'sigma_dash_t': schedule['sigma_dash_ts'][t_idx],
        }

        if t_idx > 0:
            t_1_idx = t_idx - 1
            mod_params.update({
                't_1': int(ts_array[t_1_idx]), # Store the actual t-1 value
                't_1_idx': t_1_idx,
                'alpha_t_1': schedule['alpha_ts'][t_1_idx],
                'sigma_t_1': schedule['sigma_ts'][t_1_idx],
                'alpha_dash_t_1': schedule['alpha_dash_ts'][t_1_idx],
                'var_dash_t_1': schedule['var_dash_ts'][t_1_idx],
                'sigma_dash_t_1': schedule['sigma_dash_ts'][t_1_idx],
            })
        else:
             mod_params.update({
                't_1': 0,
                't_1_idx': -1, # Indicate no t-1 index
                'alpha_t_1': 1.0, # Defaults for t=0
                'sigma_t_1': 0.0,
                'alpha_dash_t_1': 1.0,
                'var_dash_t_1': 0.0,
                'sigma_dash_t_1': 0.0,
            })

        noise_params[mod] = mod_params

    return noise_params