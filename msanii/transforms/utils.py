import torch


def get_complex_dtype(real_dtype: torch.dtype) -> torch.dtype:
    if "double" in repr(real_dtype):
        return torch.cdouble
    if "float16" in repr(real_dtype):
        return torch.complex32
    if "float" in repr(real_dtype):
        return torch.cfloat

    raise ValueError(f"Unexpected dtype {real_dtype}")
