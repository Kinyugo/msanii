import importlib
from inspect import signature
from typing import Any, Callable, Mapping, Optional


def from_config(
    config: Mapping, target: Optional[Callable] = None, **kwargs: Any
) -> Any:
    # Convert config to a dictionary so we can add more keys
    config = dict(config)

    # Add target import path if given
    if target is not None:
        target_path = get_import_path_from_instance(target)
        config.update({"_target_": target_path})

    # Get parameters of the target
    target = import_from_path(config.get("_target_"))
    target_signature = signature(target)
    target_params = [param for param in target_signature.parameters]

    # Select only arguments that are in the parameters of the target
    filtered_config = {k: v for k, v in config.items() if k in target_params}

    # Add placeholder kwargs
    if config.get("_kwargs_", None) is None:
        config.update({"_kwargs_": {}})

    # Merge the configs
    merged_config = {**filtered_config, **config.get("_kwargs_"), **kwargs}

    return target(**merged_config)


def get_import_path_from_instance(instance: Any) -> str:
    return f"{instance.__module__}.{instance.__name__}"


def import_from_path(import_path: str) -> Callable:
    module_name, obj_name = import_path.rsplit(".", maxsplit=1)
    module = importlib.import_module(module_name)

    return getattr(module, obj_name)
