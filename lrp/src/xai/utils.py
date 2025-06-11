import copy
from typing import Generator, Callable
import torch

def flatten_model(module: torch.nn.Module):
    """
    Flatten moduls to base operation like Conv2, Linear, ...
    """
    modules_list = []
    for m_1 in module.children():

        if len(list(m_1.children())) == 0:
            modules_list.append(m_1)
        else:
            modules_list = modules_list + flatten_model(m_1)
    return modules_list

def stabilize_division_by_zero(x: torch.tensor):
    """Handles potential division by zero.

    Args:
        x (torch.tensor): The input tensor

    Returns:
        torch.tensor: Returns the input tensor where zero entries were replaced. 
    """
    eps = torch.finfo(x.dtype).eps
    return x + (x == 0).to(x.dtype) * eps + torch.sign(x) * eps


def collect_leaves(module: torch.nn.Module) -> Generator[torch.nn.Module, None, None]:
    """Creates generator to collect all leaf modules of a module.

    Args:
        module (torch.nn.Module): A module for which the leaves will be collected. 
    Yields:
        Generator[torch.nn.Module, None, None]: Either a leaf of the module structure, or the module itself if it has no children.
    """
    is_leaf = True

    children = module.children()
    for child in children:
        is_leaf = False
        for leaf in collect_leaves(child):
            yield leaf
    if is_leaf:
        yield module

def modified_layer(layer: torch.nn.Module, transform: Callable[[torch.Tensor], torch.Tensor]) -> torch.nn.Module:
    """
    Deep-copy a layer and apply `transform` to its weight and bias.
    """
    new_layer = copy.deepcopy(layer)
    for name in ('weight', 'bias'):
        param = getattr(layer, name, None)
        if param is not None:
            new_param = transform(param.detach().clone())
            setattr(new_layer, name, torch.nn.Parameter(new_param))
    return new_layer