import logging
from typing import Any, Optional

import torch
import torchvision.models as tv_models

from lrp.src.trainer.base_trainer import BaseTrainer
from lrp.src.trainer.classification_trainer import ClassificationTrainer
from lrp.src.experiments.tracker import ExperimentLogger

def create_trainer_from_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: str,
    num_epochs: int,
    experiment_path: str,
    training_logger: logging.Logger,
    experiment_logger: ExperimentLogger,
    scheduler: Optional[Any] = None,
) -> BaseTrainer:
    """
    Creates a trainer instance from the provided model and trainer parameters.

    If the model is a standard torchvision backbone (ResNet, VGG, DenseNet), 
    we always dispatch to ClassificationTrainer. Otherwise, we look up the trainer 
    in BaseTrainer._registry by model.model_id.
    """

    # If model is a torchvision ResNet, VGG, or DenseNet, use the one generic ClassificationTrainer
    trainer_class = ClassificationTrainer
    
    return trainer_class(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=num_epochs,
        experiment_path=experiment_path,
        training_logger=training_logger,
        experiment_logger=experiment_logger,
        scheduler=scheduler
    )
