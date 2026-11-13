import torch


def scatter_max(src, index, dim_size):
    out = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    return out.scatter_reduce_(0, index, src, reduce="amax", include_self=False)

def scatter_min(src, index, dim_size):
    out = torch.ones(dim_size, dtype=src.dtype, device=src.device)
    return out.scatter_reduce_(0, index, src, reduce="amin", include_self=False)

def scatter_add(src, index, dim_size):
    out = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    return out.scatter_reduce_(0, index, src, reduce="sum", include_self=False)

def scatter_mul(src, index, dim_size):
    out = torch.ones(dim_size, dtype=src.dtype, device=src.device)
    return out.scatter_reduce_(0, index, src, reduce="prod", include_self=False)
