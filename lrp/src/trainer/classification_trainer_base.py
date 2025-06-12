import os
import logging
from typing import Optional, Tuple

import numpy
import torch
from torch import nn
from torch.utils.data import DataLoader

from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from lrp.src.experiments.tracker import ExperimentLogger


class BaseClassificationTrainer:
    """
    A minimal classification trainer with:
      - Rich progress bars (training & validation)
      - best-model checkpointing (based on lowest validation loss)
      - basic Python logging
      - optional LR scheduler
      - optional SimpleExperimentLogger to save (epoch, train_loss, train_acc, val_loss, val_acc) to CSV
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        num_epochs: int,
        checkpoint_dir: str,
        logger: Optional[logging.Logger] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        multi_label: bool = False,
        experiment_logger: Optional[ExperimentLogger] = None,
        class_weights: numpy.ndarray[float] = None,
    ):
        """
        Args:
            model: your torch.nn.Module
            optimizer: torch optimizer
            train_loader: DataLoader for training
            val_loader: DataLoader for validation
            device: torch.device("cpu") or torch.device("cuda")
            num_epochs: how many epochs to run
            checkpoint_dir: where to save best-model checkpoints
            logger: optional Python logger (will create one if None)
            scheduler: optional LR scheduler
            multi_label: True if using BCEWithLogitsLoss, False for CrossEntropyLoss
            experiment_logger: optional SimpleExperimentLogger instance
        """
        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.class_weights = class_weights

        # Choose loss function
        if multi_label:
            self.criterion = nn.BCEWithLogitsLoss(weight=torch.tensor(self.class_weights, dtype=torch.float32))
            self.multi_label = True
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.multi_label = False

        self.scheduler = scheduler
        self.logger = logger

        # Track best validation loss
        self.best_val_acc = 0
        self.best_model_path: Optional[str] = None

        # Lightweight experiment‐logger (optional)
        self.experiment_logger = experiment_logger

    def train_one_epoch(self, epoch: int):
        """
        Runs one epoch of training with a Rich progress bar.
        Returns (average training loss, training accuracy).
        """
        self.model.train()
        running_loss = 0.0
        running_corrects = 0
        running_total = 0

        num_batches = len(self.train_loader)

        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=False,
        ) as progress:
            task = progress.add_task(f"Epoch {epoch} TRAIN", total=num_batches)
            for batch_idx, (inputs, labels) in enumerate(self.train_loader, start=1):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                if self.multi_label:
                    loss = self.criterion(outputs, labels.float())
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    correct_bits = (preds == labels).float().sum().item()
                    # total number of bits in this batch = batch_size * num_labels
                    total_bits = labels.numel()
                    running_corrects += correct_bits
                    running_total += total_bits

                else:
                    loss = self.criterion(outputs, labels)
                    preds = outputs.argmax(dim=1)
                    correct_samples = (preds == labels).sum().item()
                    batch_size = inputs.size(0)
                    running_corrects += correct_samples
                    running_total += batch_size

                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

                avg_loss_so_far = running_loss / batch_idx
                acc_so_far = running_corrects / running_total

                progress.update(
                    task,
                    advance=1,
                    description=(
                        f"[blue]Epoch {epoch} TRAIN | "
                        f"Loss: {avg_loss_so_far:.4f} | Acc: {acc_so_far * 100:.2f}%"
                    ),
                )
        avg_train_loss = running_loss / num_batches
        train_acc = running_corrects / running_total
        self.logger.info(
            f"Epoch {epoch} completed. "
            f"Average Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc*100:.2f}%"
        )
        return avg_train_loss, train_acc

    def validate(self, epoch: int):
        """
        Runs one pass through the validation set with a Rich progress bar.
        Returns (average validation loss, validation accuracy).
        Also checkpoints if loss is the best so far.
        """
        self.model.eval()
        running_val_loss = 0.0
        running_corrects = 0
        running_total = 0

        num_batches = len(self.val_loader)

        with torch.no_grad():
            with Progress(
                TextColumn("[bold green]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                transient=False,
            ) as progress:
                task = progress.add_task(f"Epoch {epoch} VALIDATION", total=num_batches)
                for batch_idx, (inputs, labels) in enumerate(self.val_loader, start=1):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(inputs)
            
                    if self.multi_label:
                        loss = self.criterion(outputs, labels.float())
                        preds = (torch.sigmoid(outputs) > 0.5).float()
                        correct_bits = (preds == labels).float().sum().item()
                        # total number of bits in this batch = batch_size * num_labels
                        total_bits = labels.numel()
                        running_corrects += correct_bits
                        running_total += total_bits

                    else:
                        loss = self.criterion(outputs, labels)
                        preds = outputs.argmax(dim=1)
                        correct_samples = (preds == labels).sum().item()
                        batch_size = inputs.size(0)
                        running_corrects += correct_samples
                        running_total += batch_size

                    running_val_loss += loss.item()
                    avg_val_so_far = running_val_loss / batch_idx
                    acc_val_so_far = running_corrects / running_total

                    progress.update(
                        task,
                        advance=1,
                        description=(
                            f"[green]Epoch {epoch} VALIDATE | "
                            f"Loss: {avg_val_so_far:.4f} | Acc: {acc_val_so_far*100:.2f} %"
                        ),
                    )

        avg_val_loss = running_val_loss / num_batches
        val_acc = running_corrects / running_total
        self.logger.info(f"Epoch {epoch} → Validation Loss: {avg_val_loss:.4f}, Val Acc: {val_acc*100:.4f}")

        # Checkpoint if improved
        if self.best_val_acc < val_acc:
            
            for name in os.listdir(self.checkpoint_dir):
                path = os.path.join(self.checkpoint_dir, name)
                # Only remove regular files (skip directories, symlinks, etc.)
                if os.path.isfile(path):
                    os.remove(path)
                    self.logger.info(f"Deleted file: {path}")
                else:
                    self.logger.info(f"Skipped (not a file): {path}")
            self.best_val_acc = val_acc
            ckpt_name = f"best_model_epoch_{epoch}.pt"
            ckpt_path = os.path.join(self.checkpoint_dir, ckpt_name)
            torch.save(self.model.state_dict(), ckpt_path)
            self.best_model_path = ckpt_path
            self.logger.info(f"New best model saved at epoch {epoch}: {ckpt_path}")

        return avg_val_loss, val_acc

    def run(self):
        """
        Full training + validation loop. After each epoch, logs metrics to CSV
        if an Experiment Logger is provided. Steps scheduler if any.
        """
        for epoch in range(1, self.num_epochs + 1):
            self.logger.info(f"Starting Epoch {epoch}/{self.num_epochs}")

            # Training phase
            train_loss, train_acc = self.train_one_epoch(epoch)

            # Validatation phase
            val_loss, val_acc = self.validate(epoch)

            # Record metrics if experiment_logger is set
            if self.experiment_logger is not None:
                self.experiment_logger.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc)

            # Set Step scheduler
            if self.scheduler is not None:
                try:
                    # ReduceLROnPlateau style
                    self.scheduler.step(val_loss)
                    self.logger.info("Scheduler.step(val_loss) called.")
                except TypeError:
                    # Any other scheduler 
                    self.scheduler.step()
                    self.logger.info("Scheduler.step() called.")

        self.logger.info("Training complete.")
        if self.best_model_path:
            self.logger.info(f"Best model checkpoint: {self.best_model_path}")
        else:
            self.logger.info("No checkpoint was saved.")