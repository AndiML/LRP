import torch
import torch.nn as nn

# A global registry to hold activations and processed weights for each module
_activation_store = {}
_weight_store     = {}

def _linear_forward_hook(module, inputs, outputs):
    """
    This forward hook will run for every nn.Linear (or similar) 
    to store:
      - the input activation a^{(l-1)}
      - the original weight tensor W
    """
    # inputs is a tuple; for Linear it’s (x,) where x has shape [batch, in_features].
    x = inputs[0].detach()                      
    W = module.weight.detach()             
    b = module.bias.detach() if (module.bias is not None) else None

    # stash them by module id
    _activation_store[module] = x
    _weight_store[module]     = (W, b)


def register_forward_hooks(model):
    """
    Walks the model, and for each nn.Linear or nn.Conv2d (or custom),
    registers the _linear_forward_hook so we can grab a^(l-1) and W^(l).
    """
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.register_forward_hook(_linear_forward_hook)



