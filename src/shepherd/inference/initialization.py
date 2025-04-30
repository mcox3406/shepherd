import torch
import numpy as np
import torch_scatter


# helper function to initialize x1 state
def _initialize_x1_state(batch_size, N_x1, params, prior_noise_scale, include_virtual_node):
    num_atom_types = len(params['dataset']['x1']['atom_types']) + len(params['dataset']['x1']['charge_types'])
    
    bond_adj = np.triu(1-np.diag(np.ones(N_x1, dtype = int)))
    bond_edge_index = np.stack(bond_adj.nonzero(), axis = 0)
    bond_edge_index = bond_edge_index + int(include_virtual_node)
    bond_edge_index = torch.as_tensor(bond_edge_index, dtype = torch.long)
    
    x1_batch = torch.cat([
        torch.ones(N_x1 + int(include_virtual_node), dtype = torch.long) * i for i in range(batch_size)
    ])
    virtual_node_pos_x1 = torch.tensor([[0.,0.,0.]], dtype = torch.float)
    virtual_node_x_x1 = torch.zeros(num_atom_types, dtype = torch.float)
    virtual_node_x_x1[0] = 1. * params['dataset']['x1']['scale_atom_features']
    virtual_node_x_x1 = virtual_node_x_x1[None, ...]
    
    virtual_node_mask_x1 = []
    pos_forward_noised_x1 = []
    x_forward_noised_x1 = []
    bond_edge_x_forward_noised_x1 = [] 
    bond_edge_index_x1 = []
    num_nodes_counter = 0
    for b in range(batch_size):
        x1_pos_T = torch.randn(N_x1, 3) * prior_noise_scale
        x1_pos_T = x1_pos_T - torch.mean(x1_pos_T, dim = 0)
        x1_x_T = torch.randn(N_x1, num_atom_types)
        x1_bond_edge_x_T = torch.randn(bond_edge_index.shape[1], len(params['dataset']['x1']['bond_types']))
        
        x1_virtual_node_mask_ = torch.zeros(N_x1 + int(include_virtual_node), dtype = torch.long)
        if include_virtual_node:
            x1_virtual_node_mask_[0] = 1
            x1_pos_T = torch.cat([virtual_node_pos_x1, x1_pos_T], dim = 0)
            x1_x_T = torch.cat([virtual_node_x_x1, x1_x_T], dim = 0)
        x1_virtual_node_mask_ = x1_virtual_node_mask_ == 1

        pos_forward_noised_x1.append(x1_pos_T)
        x_forward_noised_x1.append(x1_x_T)
        virtual_node_mask_x1.append(x1_virtual_node_mask_)
        bond_edge_x_forward_noised_x1.append(x1_bond_edge_x_T)
        bond_edge_index_x1.append(bond_edge_index + num_nodes_counter)
        num_nodes_counter += N_x1 + int(include_virtual_node)
        
    pos_forward_noised_x1 = torch.cat(pos_forward_noised_x1, dim = 0)
    x_forward_noised_x1 = torch.cat(x_forward_noised_x1, dim = 0)
    virtual_node_mask_x1 = torch.cat(virtual_node_mask_x1, dim =0)
    bond_edge_x_forward_noised_x1 = torch.cat(bond_edge_x_forward_noised_x1, dim =0)
    bond_edge_index_x1 = torch.cat(bond_edge_index_x1, dim =1)
    
    return (pos_forward_noised_x1, x_forward_noised_x1, bond_edge_x_forward_noised_x1, 
            x1_batch, virtual_node_mask_x1, bond_edge_index_x1)


# helper function to initialize x2 state
def _initialize_x2_state(batch_size, N_x2, params, prior_noise_scale, include_virtual_node):
    x2_batch = torch.cat([
        torch.ones(N_x2 + int(include_virtual_node), dtype = torch.long) * i for i in range(batch_size)
    ])
    virtual_node_pos_x2 = torch.tensor([[0.,0.,0.]], dtype = torch.float) # same as virtual node for x1

    virtual_node_mask_x2 = []
    pos_forward_noised_x2 = []
    x_forward_noised_x2 = [] # this is an unnoised one-hot embedding of real/virtual node
    for b in range(batch_size):
        x2_x_T = torch.zeros((N_x2 + int(include_virtual_node), 2))
        x2_x_T[:,0] = 1
        if include_virtual_node:
            x2_x_T[0,0] = 0
            x2_x_T[0,1] = 1
        
        x2_pos_T = torch.randn(N_x2, 3) * prior_noise_scale # t = T 
        # NOT removing COM
    
        x2_virtual_node_mask_ = torch.zeros(N_x2 + int(include_virtual_node), dtype = torch.long)
        if include_virtual_node:
            x2_virtual_node_mask_[0] = 1
            x2_pos_T = torch.cat([virtual_node_pos_x2, x2_pos_T], dim = 0)
        x2_virtual_node_mask_ = x2_virtual_node_mask_ == 1
        
        pos_forward_noised_x2.append(x2_pos_T)
        x_forward_noised_x2.append(x2_x_T)
        virtual_node_mask_x2.append(x2_virtual_node_mask_)
        
    pos_forward_noised_x2 = torch.cat(pos_forward_noised_x2, dim = 0)
    x_forward_noised_x2 = torch.cat(x_forward_noised_x2, dim = 0)
    virtual_node_mask_x2 = torch.cat(virtual_node_mask_x2, dim =0)

    return pos_forward_noised_x2, x_forward_noised_x2, x2_batch, virtual_node_mask_x2


# helper function to initialize x3 state
def _initialize_x3_state(batch_size, N_x3, params, prior_noise_scale, include_virtual_node):
    x3_batch = torch.cat([
        torch.ones(N_x3 + int(include_virtual_node), dtype = torch.long) * i for i in range(batch_size)
    ])
    virtual_node_pos_x3 = torch.tensor([[0.,0.,0.]], dtype = torch.float) # same as virtual node for x1
    virtual_node_x_x3 = torch.tensor([0.0], dtype = torch.float)

    virtual_node_mask_x3 = []
    pos_forward_noised_x3 = []
    x_forward_noised_x3 = []
    for b in range(batch_size):
        x3_x_T = torch.randn(N_x3, dtype = torch.float) * prior_noise_scale
        x3_pos_T = torch.randn(N_x3, 3) * prior_noise_scale # t = T 
        # NOT removing COM from x2/x3 starting structure
    
        x3_virtual_node_mask_ = torch.zeros(N_x3 + int(include_virtual_node), dtype = torch.long)
        if include_virtual_node:
            x3_virtual_node_mask_[0] = 1
            x3_pos_T = torch.cat([virtual_node_pos_x3, x3_pos_T], dim = 0)
            x3_x_T = torch.cat([virtual_node_x_x3, x3_x_T], dim = 0)
        x3_virtual_node_mask_ = x3_virtual_node_mask_ == 1
        
        pos_forward_noised_x3.append(x3_pos_T)
        x_forward_noised_x3.append(x3_x_T)
        virtual_node_mask_x3.append(x3_virtual_node_mask_)
        
    pos_forward_noised_x3 = torch.cat(pos_forward_noised_x3, dim = 0)
    x_forward_noised_x3 = torch.cat(x_forward_noised_x3, dim = 0)
    virtual_node_mask_x3 = torch.cat(virtual_node_mask_x3, dim =0)

    return pos_forward_noised_x3, x_forward_noised_x3, x3_batch, virtual_node_mask_x3


# helper function to initialize x4 state
def _initialize_x4_state(batch_size, N_x4, params, prior_noise_scale, include_virtual_node):
    num_pharm_types = params['dataset']['x4']['max_node_types']

    x4_batch = torch.cat([
        torch.ones(N_x4 + int(include_virtual_node), dtype = torch.long) * i for i in range(batch_size)
    ])
    virtual_node_direction_x4 = torch.tensor([[0.,0.,0.]], dtype = torch.float) 
    virtual_node_pos_x4 = torch.tensor([[0.,0.,0.]], dtype = torch.float) # same as virtual node for x1
    virtual_node_x_x4 =  torch.zeros(num_pharm_types, dtype = torch.float)
    virtual_node_x_x4[0] = 1. * params['dataset']['x4']['scale_node_features'] # one-hot encoding, that remains unnoised
    virtual_node_x_x4 = virtual_node_x_x4[None, ...]

    virtual_node_mask_x4 = []
    pos_forward_noised_x4 = []
    direction_forward_noised_x4 = []
    x_forward_noised_x4 = []
    for b in range(batch_size):
        x4_pos_T = torch.randn(N_x4, 3) * prior_noise_scale # t = T
        # NOT removing COM from x4
        x4_direction_T = torch.randn(N_x4, 3) * prior_noise_scale # t = T
        x4_x_T = torch.randn(N_x4, num_pharm_types)
        
        x4_virtual_node_mask_ = torch.zeros(N_x4 + int(include_virtual_node), dtype = torch.long)
        if include_virtual_node:
            x4_virtual_node_mask_[0] = 1
            x4_pos_T = torch.cat([virtual_node_pos_x4, x4_pos_T], dim = 0)
            x4_direction_T = torch.cat([virtual_node_direction_x4, x4_direction_T], dim = 0)
            x4_x_T = torch.cat([virtual_node_x_x4, x4_x_T], dim = 0)
        x4_virtual_node_mask_ = x4_virtual_node_mask_ == 1

        pos_forward_noised_x4.append(x4_pos_T)
        direction_forward_noised_x4.append(x4_direction_T)
        x_forward_noised_x4.append(x4_x_T)
        virtual_node_mask_x4.append(x4_virtual_node_mask_)
        
    pos_forward_noised_x4 = torch.cat(pos_forward_noised_x4, dim = 0)
    direction_forward_noised_x4 = torch.cat(direction_forward_noised_x4, dim = 0)
    x_forward_noised_x4 = torch.cat(x_forward_noised_x4, dim = 0)
    virtual_node_mask_x4 = torch.cat(virtual_node_mask_x4, dim =0)

    return (pos_forward_noised_x4, direction_forward_noised_x4, x_forward_noised_x4, 
            x4_batch, virtual_node_mask_x4)