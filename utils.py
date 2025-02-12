import torch.nn as nn
import argparse
import os

def count_parameters(model:nn.Module)->int:
    """
    Count the number of parameters in a model.
    
    Args:
    - model: nn.Module, the model to count parameters.
    
    Returns:
    - int, the number of parameters in the model.
    """
    num_params =  sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Number of parameters: {num_params}")
    return num_params


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def ensure_dir(dir_name: str):
    """Creates folder if not exists.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)



