import torch
from lrp_base import AbstractLrpRule


class LrpZeroRule(AbstractLrpRule):
    """
    LRP-0 rule (zero-rule).

    The simplest LRP rule: no epsilon or gamma modifications.
    Denominator z is exactly the layer’s output.
    """
    rule_id = "zero-rule"

    def __init__(self, layer: torch.nn.Module):
        """
        Initialize the LRP-0 rule wrapper.

        Args:
            layer (torch.nn.Module): The layer to wrap with LRP-0 rule.
        """
        super().__init__(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the LRP-0 rule for relevance propagation.

        Args:
            x (torch.Tensor): Input activations to this layer.

        Returns:
            torch.Tensor: LRP-modified output, ready for backward relevance propagation.
        """
        out = self.layer(x)
        z   = out
        return self.modified_forward(z, out)

class LrpEpsilonRule(AbstractLrpRule):
    """
    LRP-ε rule (epsilon-rule).

    Adds a small stabilizing epsilon to the denominator to avoid division by zero.
    """
    rule_id = "epsilon-rule"

    def __init__(self, layer: torch.nn.Module, epsilon: float):
        """
        Initialize the LRP-ε rule wrapper.

        Args:
            layer (torch.nn.Module): The layer to wrap with LRP-ε rule.
            epsilon (float): Stabilizing constant ε added to denominator.
        """
        super().__init__(layer)
        self.epsilon = epsilon
        # make copy of layer for computing z
        self._make_copy("copy_layer", lambda p: p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the LRP-ε rule for relevance propagation.

        Args:
            x (torch.Tensor): Input activations to this layer.

        Returns:
            torch.Tensor: LRP-modified output, ready for backward relevance propagation.
        """
        out = self.layer(x)
        z_copy = self.copy_layer(x)
        z = self.epsilon + z_copy

        return self.modified_forward(z, out)


class LrpGammaRule(AbstractLrpRule):
    """
    LRP-gamma rule (gamma-rule).

    Boosts positive contributions by adding gamma times positive-only activations.
    """
    rule_id = "gamma-rule"

    def __init__(self, layer: torch.nn.Module, gamma: float):
        """
        Initialize the LRP-gamma rule wrapper.

        Args:
            layer (torch.nn.Module): The layer to wrap with LRP-gamma rule.
            gamma (float): Weighting factor for positive contributions.
        """
        super().__init__(layer)
        
        self.gamma = gamma
        # make copy of layer with modified params
        self._make_copy("copy_layer", lambda p: p + gamma * p.clamp(min=0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the LRP-gamma rule for relevance propagation.

        Args:
            x (torch.Tensor): Input activations to this layer.

        Returns:
            torch.Tensor: LRP-modified output, ready for backward relevance propagation.
        """
        out    = self.layer(x)
        z_copy = self.copy_layer(x)
        z      = z_copy
        
        return out #self.modified_forward(z, out)

class LrpZBoxRule(AbstractLrpRule):
    """
    LRP-ZBox rule (zbox-rule).

    Uses separate low/high parameter copies to bound relevance propagation.
    """
    rule_id = "zbox-rule"

    def __init__(self, layer: torch.nn.Module, low: torch.Tensor, high: torch.Tensor):
        """
        Initialize the LRP-ZBox rule wrapper.

        Args:
            layer (torch.nn.Module): The layer to wrap with LRP-ZBox rule.
            low (torch.Tensor): Lower-bound tensor for input range.
            high (torch.Tensor): Upper-bound tensor for input range.
        """
        super().__init__(layer)
        
        self._make_copy("low_layer",  lambda p: p.clamp(min=0))
        self._make_copy("high_layer", lambda p: p.clamp(max=0))
        
        self.low  = low.clone().requires_grad_(True)
        self.high = high.clone().requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the LRP-ZBox rule for relevance propagation.

        Args:
            x (torch.Tensor): Input activations to this layer.

        Returns:
            torch.Tensor: LRP-modified output, ready for backward relevance propagation.
        """
        out    = self.layer(x)
        z_low  = self.low_layer(self.low)
        z_high = self.high_layer(self.high)
        z      = out - z_low - z_high
        
        return self.modified_forward(z, out)
