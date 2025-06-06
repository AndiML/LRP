"""Represents a module that contains experiment management utilities."""

import os
import csv
import yaml
import logging
from datetime import datetime
from argparse import Namespace
from typing import Any, Optional, Dict

from torch.utils.tensorboard.writer import SummaryWriter

class ExperimentLogger:
    """
    A versatile experiment logger that supports:
      - Saving hyperparameters (YAML).
      - Logging per-epoch training metrics to a CSV file.
      - Logging metrics to TensorBoard (if enabled).
      - Automatic handling of model-specific metrics based on a model tag and task type.
    """

    def __init__(
        self,
        output_path: str,
        logger: logging.Logger,
        use_tensorboard: bool = False,
    ) -> None:
        """
        Initializes the experiment logger.

        Args:
            output_path (str): Directory where logs will be saved.
            task_type (str): Type of task (e.g., 'reconstruction', 'classification').
            logger(logging.Logger): Logger is provided to log metrics obtained during the training process directly to the command line.
            use_tensorboard (bool): If True, TensorBoard logging is enabled.

        """
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
        self.logger = logger
        self.use_tensorboard = use_tensorboard

        # Set up CSV logging.
        self.metrics_file_path = os.path.join(self.output_path, 'metrics.csv')
        self.metrics_file = open(self.metrics_file_path, 'w', encoding='utf-8', newline='')
        self.csv_writer = csv.writer(self.metrics_file)
        self.csv_writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        self.metrics_file.flush()

        # Initializes TensorBoard logging if enabled.
        if self.use_tensorboard:
            self.tensorboard_writer = SummaryWriter(log_dir=self.output_path)
        else:
            self.tensorboard_writer = None


    def display_hyperparamter_for_training(self, command_line_arguments: Namespace) -> None:
        """Displays the parameters that are used during the training.

        Args:
            command_line_arguments (Namespace): The command line arguments that are used in the training process over the in distribution data.

        """
        # Build a dictionary of the most interesting hyperparameters.
        hyperparams = {
            "Model": command_line_arguments.model_type.upper(),
            "Optimizer": command_line_arguments.optimizer.upper(),
            "Scheduler": command_line_arguments.scheduler.upper(),
            "Learning Rate": command_line_arguments.learning_rate,
            "Epochs": command_line_arguments.epochs,
            "Batch Size": command_line_arguments.batchsize,
            "Momentum": command_line_arguments.momentum,
            "Weight Decay": command_line_arguments.weight_decay,
            "Use GPU": command_line_arguments.use_gpu,
        }

        self.logger.info("\nExperimental Details:")
        for key, value in hyperparams.items():
            # Use formatting for neat alignment.
            self.logger.info(f"[blue]{key:20s}: {value}[/blue]")

    def save_hyperparameters(self, hyperparameters: Dict[str, Any]) -> None:
        """
        Saves hyperparameters to a YAML file.

        Args:
            hyperparameters (Dict[str, Any]): A dictionary of hyperparameter names and values.
        """
        hp_path = os.path.join(self.output_path, 'hyperparameters.yaml')
        with open(hp_path, 'w', encoding='utf-8') as f:
            yaml.dump(hyperparameters, f)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
    ) -> None:
        """
        Appends a row: [epoch, train_loss, train_acc, val_loss, val_acc] to the CSV.
        """
        self.csv_writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])
        self.metrics_file.flush()
        self.logger.info(
            f"[ExperimentLogger] Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

    def close(self) -> None:
        """
        Closes file handles and the TensorBoard writer.
        """
        if not self.metrics_file.closed:
            self.metrics_file.close()
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()

