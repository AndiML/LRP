import torch
from abc import ABC, abstractmethod
from typing import Dict, Type, Callable
from utils import modified_layer, stabilize_division_by_zero

class AbstractLrpRule(torch.nn.Module, ABC):
    """
    Base class for LRP rules with automatic registry.

    Subclasses should define a class-level `rule_id` string.
    The registry maps rule_id -> subclass for dynamic lookup.
    """
    # Registry to automatically register subclasses by rule_id
    _registry: Dict[str, Type['AbstractLrpRule']] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, 'rule_id'):
            rid = getattr(cls, 'rule_id')
            cls._registry[rid] = cls

    def __init__(self, layer: torch.nn.Module):
        super().__init__()
        self.layer = layer

    def _make_copy(self, name: str, transform: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """
        Create and attach a modified copy of the wrapped layer.

        This helper deep-copies the original layer stored in self.layer, applies the
        provided transform function to each of its parameters (weight, bias),
        and assigns the new layer under `self.<name>` for later use in forward.

        Args:
            name (str):  Attribute name to assign the new layer.
            transform (Callable):  Function applied to each parameter tensor to modify it.
        """
        new_layer = modified_layer(self.layer, transform)
        setattr(self, name, new_layer)
    
    def modified_forward(self, z: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        Apply the shared LRP gradient modification.

        Computes the element-wise product of z and the detached ratio
        output / stabilize_division_by_zero(z), so that during backpropagation
        only z contributes gradients, implementing the LRP redistribution step:

            R = z * (output / z).detach()

        Args:
            z (torch.Tensor):  Numerator tensor for relevance (layer-specific denominator).
            output (torch.Tensor):  Original forward activation of the layer.

        Returns:
            torch.Tensor:  LRP-modified tensor, ready for backward pass.
        """
        return z * (output / stabilize_division_by_zero(z)).detach()

    @abstractmethod
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Perform the layer-wise LRP forward pass.

        This method should compute the LRP-adjusted output for the wrapped layer,
        given its input activations. It typically:
        1. Runs the original forward to get activations out = self.layer(input_tensor).
        2. Constructs a modified denominator z (e.g. adding ε, applying gamma-rule, etc.).
        3. Calls self.modified_forward(z, out) to produce the relevance-aware tensor.

        Args:
            input_tensor (torch.Tensor):  The input activations to this layer.

        Raises:
            NotImplementedError:  Must be overridden by each concrete LRP rule subclass.

        Returns:
            torch.Tensor:  The relevance-propagated output for this layer.
        """
        raise NotImplementedError