from lrp.src.models.base_model import BaseModel

def create_model(model_kind: str = None, **kwargs) -> BaseModel:
    """Instantiate a model by looking up registry of the BaseModel. """
    
    if model_kind is None:
        model_kind = DEFAULT_MODEL_ID

    if model_kind not in BaseModel._registry:
        raise ValueError(f"Unknown model_kind '{model_kind}'.")

    model_cls = BaseModel._registry[model_kind]
    return model_cls(**kwargs)