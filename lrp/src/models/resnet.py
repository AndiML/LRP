import torch
import torchvision

from lrp.src.models.base_model import BaseModel

class ResNetModel(BaseModel):
    model_id = "resnet"

    def __init__(
        self,
        version: str = "resnet50",
        num_targets: int = 40,
        pretrained: bool = True,
    ) -> None:
        """
        Args:
            version (str): Name of a torchvision ResNet constructor, e.g. "resnet34", "resnet50", etc. Must match exactly a function in torchvision.models (otherwise raises ValueError).
            num_targets (int): Number of output units (e.g. # of face attributes or classes).
            pretrained (bool): If True, load ImageNet-pretrained weights.
        """
        super().__init__()

        # Attempt to grab torchvision.models.<version> dynamically.
        # This will succeed for "resnet34", "resnet50", "resnet101", etc.
        try:
            model_fn = getattr(torchvision.models, version)
        except AttributeError:
            raise ValueError(
                f"ResNetModel: '{version}' is not found in torchvision.models. "
                f"Valid ResNet versions include 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', etc."
            )

        candidates = [name for name in dir(torchvision.models) if name.endswith("_Weights") and name.lower().startswith(version.lower())]
        if not candidates:
            raise RuntimeError(f"No Weights enum found for '{version}'")
        if len(candidates) > 1:
            raise RuntimeError(f"Ambiguous Weights enums {candidates!r} for '{version}'")

        weights_enum = getattr(torchvision.models, candidates[0])
        weights = weights_enum.DEFAULT if pretrained else None
        backbone = model_fn(weights=weights)

        # Replace the final fully‐connected layer so that out_features = num_targets
        in_features = backbone.fc.in_features
        backbone.fc = torch.nn.Linear(in_features, num_targets)

        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
