import torch


def get_complex_dtype(real_dtype: torch.dtype) -> torch.dtype:
    if real_dtype == torch.double:
        return torch.cdouble
    if real_dtype == torch.float:
        return torch.cfloat
    if real_dtype == torch.half:
        return torch.complex32
    raise ValueError(f"Unexpected dtype {real_dtype}")
