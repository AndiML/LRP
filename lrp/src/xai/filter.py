import copy
import torch
from typing import List, Tuple, Dict, Union, Optional, Type, Callable
from lrp_base import AbstractLrpRule
from torchvision.models import vgg16
from lrp_rules import LrpZeroRule, LrpEpsilonRule, LrpGammaRule

def _rsetattr(obj: torch.nn.Module, attr: str, val: torch.nn.Module) -> None:
    """
    Recursively set an attribute on a module given a dotted path.
    """
    parts = attr.split('.')
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], val)


def filter_by_layer_index_type(
    fn: Callable[[int], bool],
    mod_type: Optional[Type[torch.nn.Module]] = None
) -> Callable[[str, torch.nn.Module, int], bool]:
    """
    Returns a filter function matching layers by index and optional module type.

    Args:
        fn: Function taking layer index and returning bool.
        mod_type: If provided, also require isinstance(layer, mod_type).
    """
    def _filter(name: str, layer: torch.nn.Module, idx: int) -> bool:
        if not fn(idx):
            return False
        if mod_type is not None and not isinstance(layer, mod_type):
            return False
        return True
    return _filter


class LRPModel(torch.nn.Module):
    """Wraps a PyTorch model for Layer-wise Relevance Propagation (LRP) with flexible mapping."""

    def __init__(
        self,
        model: torch.nn.Module,
        rule_layer_map: List[
            Tuple[
                # list of keys: names, types, or filter functions
                List[Union[str, Type[torch.nn.Module], Callable[[str, torch.nn.Module, int], bool]]],
                # LRP rule id
                str,
                # parameters for the rule
                Dict[str, Union[torch.Tensor, float]]
            ]
        ],
        top_k: float = 0.0
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.model = model.eval()
        self.rule_layer_map = rule_layer_map

        # Apply LRP wrappers
        self._convert_layers() 

    def _convert_layers(self) -> None:
        """
        Traverse named modules with indices, and replace eligible layers with LRP-wrapped layers.
        """
        modules = list(self.model.named_modules())
        for idx, (name, layer) in enumerate(modules):
            rule_spec = self._get_rule_for_layer(name, layer, idx)
            if rule_spec is None:
                continue
            rule_id, rule_kwargs = rule_spec
            wrapper_cls = AbstractLrpRule._registry.get(rule_id)
            if wrapper_cls is None:
                raise KeyError(f"LRP rule '{rule_id}' not found in registry.")
            wrapped = wrapper_cls(copy.deepcopy(layer), **rule_kwargs)
            _rsetattr(self.model, name, wrapped)

    def _get_rule_for_layer(
        self,
        name: str,
        layer: torch.nn.Module,
        idx: int
    ) -> Optional[Tuple[str, Dict[str, Union[torch.Tensor, float]]]]:
        """
        Return (rule_id, params) for a given layer if it matches the mapping, else None.
        Mapping entries accept module full names, types, or custom filters.
        """
        for keys, rule_id, params in self.rule_layer_map:
            for key in keys:
                # Match by module name
                if isinstance(key, str) and key == name:
                    return rule_id, params
                # Match by module type
                if isinstance(key, type) and isinstance(layer, key):
                    return rule_id, params
                # Custom filter: fn(name, layer, idx)
                if callable(key):
                    try:
                        if key(name, layer, idx):
                            return rule_id, params
                    except TypeError:
                        # skip filters expecting different signature
                        pass
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the standard model forward with LRP-wrapped layers.
        """
        return self.model(x)

# Example usage:
# low and high are torch.Tensor bounds for ZBoxRule

shape: Tuple[int] = (1, 3, 224, 224)

low: torch.Tensor = torch.zeros(*shape)
high: torch.Tensor = torch.ones(*shape)
rule_map = [
     ([filter_by_layer_index_type(lambda i: i == 0)], 'zbox-rule', {'low': low, 'high': high}),
     ([filter_by_layer_index_type(lambda i: 1 <= i <= 16, torch.nn.Conv2d)], 'gamma-rule', {'gamma': 0.25}),
     ([filter_by_layer_index_type(lambda i: 17 <= i <= 30)], 'epsilon-rule', {'epsilon': 0.25}),
     ([filter_by_layer_index_type(lambda i: i >= 31)], 'zero-rule', {}),
]
lrp_model = LRPModel(vgg16(), rule_map)

print(lrp_model)
