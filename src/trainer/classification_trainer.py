import torch
import logging
from lrp.src.trainer.base_trainer import BaseTrainer
from lrp.src.experiments.tracker import ExperimentLogger

class ClassificationTrainer(BaseTrainer):
    """
    A trainer for plain cross-entropy (or BCEWithLogits) classification.
    """

    trainer_id = "classification" 

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        num_epochs: int,
        experiment_path: str,
        training_logger: logging.Logger,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        experiment_logger: ExperimentLogger = None,
        multi_label: bool = True,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=num_epochs,
            experiment_path=experiment_path,
            training_logger=training_logger,
            scheduler=scheduler,
            experiment_logger=experiment_logger,
        )

        self.multi_label = multi_label
        # Choose correct loss:
        if self.multi_label:
            self.criterion = torch.nn.BCEWithLogitsLoss()
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

    def train_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> dict[str, float]:
        """
        1) Unpack batch
        2) Forward pass
        3) Compute loss
        4) Backward + optimizer.step() is done in BaseTrainer.train() after we return
        5) Compute any metric (e.g. accuracy) and return a dict of floats for logging.
        """
        inputs, labels = batch
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        # Zero the gradients here (BaseTrainer.train() calls optimizer.step() AFTER train_step)
        self.optimizer.zero_grad()

        logits = self.model(inputs)

        if self.multi_label:
            loss = self.criterion(logits, labels.float())
        else:
            # labels: shape (batch_size,), logits: shape (batch_size, num_classes)
            loss = self.criterion(logits, labels)

        # Backprop only after we return: BaseTrainer.train() will call loss.backward(), optimizer.step()
        loss.backward()

        return {"loss": loss.item()}

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> dict[str, float]:
        """
        Exactly like train_step but *no* backward. We sum up loss & accuracy so we get an average at the end.
        """
        inputs, labels = batch
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        logits = self.model(inputs)
        if self.multi_label:
            loss = self.criterion(logits, labels.float())
        else:
            loss = self.criterion(logits, labels)

        return {"loss": loss.item()}

    def post_validation(
        self,
        last_batch: tuple[torch.Tensor, torch.Tensor],
        epoch: int
    ):
        """
        If you want to do something after validation—e.g., visualize a few misclassified images,
        produce a confusion matrix for the last batch, etc.—you can do it here. For a minimal trainer,
        you can leave it as “pass.”
        """
        # Example (optional): log a few predictions visually to TensorBoard, etc. For now, do nothing.
        pass