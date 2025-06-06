import pkgutil
import importlib
from pathlib import Path
import torchvision

from lrp.src.models.base_model import BaseModel

def _make_variant_subclass(base_cls, variant_name: str):
    """
    Create a subclass of base_cls whose `model_id` is variant_name.
    """
    class Variant(base_cls):
        model_id = variant_name

        def __init__(self, *args, **kwargs):
            # Forces version to be our variant_name
            super().__init__(version=variant_name, *args, **kwargs)

    Variant.__name__ = f"{base_cls.__name__}_{variant_name}"
    return Variant

# Insert all models in the registry of the base model
package_dir = Path(__file__).parent
for finder, module_name, is_pkg in pkgutil.walk_packages(
    path=[str(package_dir)],
    prefix=__name__ + ".",
):
    if module_name.endswith("__init__"):
        continue
    importlib.import_module(module_name)

# Dynamically create one “variant‐specific” subclass for each torchvision.models function
# If ResNetModel is registered, then register each torchvision‐side resnetXXX as a new subclass.
if "resnet" in BaseModel._registry:
    for name, fn in torchvision.models.__dict__.items():
        if name.startswith("resnet") and callable(fn):
            if name == "resnet":
                continue
            VariantCls = _make_variant_subclass(BaseModel._registry["resnet"], name)
            BaseModel._registry[name] = VariantCls

# If VGGModel is registered, then register each torchvision‐side vggXXX as a new subclass.
if "vgg" in BaseModel._registry:
    for name, fn in torchvision.models.__dict__.items():
        if name.startswith("vgg") and callable(fn):
            if name == "vgg":
                continue
            VariantCls = _make_variant_subclass(BaseModel._registry["vgg"], name)
            BaseModel._registry[name] = VariantCls

# If DensNet is registered, then register each torchvision‐side vggXXX as a new subclass.
if "densenet" in BaseModel._registry:
    for name, fn in torchvision.models.__dict__.items():
        if name.startswith("densenet") and callable(fn):
            if name == "densenet":
                continue
            VariantCls = _make_variant_subclass(BaseModel._registry["densenet"], name)
            BaseModel._registry[name] = VariantCls

# Exposes model ids to the outside
MODEL_IDS = sorted(BaseModel._registry.keys())
DEFAULT_MODEL_ID = MODEL_IDS[0]

__all__ = [
    "MODEL_IDS",
    "DEFAULT_MODEL_ID",
]
