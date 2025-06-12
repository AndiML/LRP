import torch
import torchvision

from lrp.src.models.base_model import BaseModel

class VGGModel(BaseModel):
    model_id = "vgg"

    def __init__(
        self,
        version: str = "vgg16",
        num_targets: int = 10,
        pretrained: bool = True,
    ) -> None:
        """
        Args:
            version (str):
                Name of a torchvision VGG constructor, e.g. "vgg11", "vgg13", "vgg16", "vgg19", or their "_bn" variants like "vgg16_bn". Must match exactly a function in torchvision.models.
            num_targets (int): Number of output units (e.g. # of face attributes or classes).
            pretrained (bool): If True, load ImageNet-pretrained weights.
        """
        super().__init__()

        # Dynamically retrieve torchvision.models.<version>
        try:
            model_fn = getattr(torchvision.models, version)
        except AttributeError:
            raise ValueError("VGGModel: '{version}' is not found in torchvision.models. "
                f"Valid VGG versions include 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', "
                f"'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', etc."
            )

        # Instantiate the chosen VGG backbone
        # for e.g. version="vgg16_bn" → enum_name="VGG16_BN_Weights"
        enum_name = version.upper() + "_Weights"
        weights_enum = getattr(torchvision.models, enum_name)
        # pick pretrained or random
        weights = weights_enum.DEFAULT if pretrained else None
        backbone = model_fn(weights=weights)

        # Replaces only the final Linear layer so out_features = num_targets.
        in_features = backbone.classifier[-1].in_features
        backbone.classifier[-1] = torch.nn.Linear(in_features, num_targets)

        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
