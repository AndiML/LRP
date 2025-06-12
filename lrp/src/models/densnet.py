import torch
import torchvision

from lrp.src.models.base_model import BaseModel

class DenseNetModel(BaseModel):
    
    model_id = "densenet"
    def __init__(
        self,
        version: str = "densenet121",
        num_targets: int = 10,
        pretrained: bool = True,
    ) -> None:
        """
        Args:
            version (str): Name of a torchvision DenseNet constructor, e.g. "densenet121", "densenet169", "densenet201", or "densenet161". Must match exactly a function in torchvision.models.
            num_targets (int): Number of output units (e.g. # of face attributes or classes).
            pretrained (bool): If True, load ImageNet-pretrained weights.
        """
        super().__init__()

        # Dynamically retrieve torchvision.models.<version>
        try:
            model_fn = getattr(torchvision.models, version)
        except AttributeError:
            raise ValueError(
                f"DenseNetModel: '{version}' is not found in torchvision.models. "
                f"Valid DenseNet versions include 'densenet121', 'densenet169', "
                f"'densenet201', 'densenet161', etc."
            )
        candidates = [name for name in dir(torchvision.models) if name.endswith("_Weights") and name.lower().startswith(version.lower())]
        if not candidates:
            raise RuntimeError(f"No Weights enum found for '{version}'")
        if len(candidates) > 1:
            raise RuntimeError(f"Ambiguous Weights enums {candidates!r} for '{version}'")

        weights_enum = getattr(torchvision.models, candidates[0])
        weights = weights_enum.DEFAULT if pretrained else None
        backbone = model_fn(weights=weights)
        in_features = backbone.classifier.in_features
        backbone.classifier = torch.nn.Linear(in_features, num_targets)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)