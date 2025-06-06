import os
import logging
from datetime import datetime
from argparse import Namespace
import torch

from lrp.commands.base import BaseCommand
from lrp.src.datasets.dataset import Dataset
from lrp.src.experiments.tracker import ExperimentLogger
from lrp.src.models.model_generator import create_model
from lrp.src.trainer.classification_trainer_base import BaseClassificationTrainer
from lrp.src.training_config.configurator import get_optimizer, get_scheduler


class RunLrpCommand(BaseCommand):
    """Represents a command that represents the RunLrp command."""

    def __init__(self) -> None:
        """Initializes a new RunLrp instance. """

        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)

    def run(self, command_line_arguments: Namespace) -> None:
        """Runs the command.

        Args:
            command_line_arguments (Namespace): The parsed command line arguments.
        """

         # Log dataset download and configuration.
        self.logger.info("Loeading Dataset for LRP Experiments: %s",command_line_arguments.dataset.upper())
        dataclass_instance = Dataset.create(command_line_arguments.dataset, command_line_arguments.dataset_path)

        print("Is MPS available?", torch.backends.mps.is_available())

        # Configure device.
        device = 'mps' if command_line_arguments.use_gpu else 'cpu'
        self.logger.info("Using device: %s for training on dataset: %s", device.upper(), command_line_arguments.dataset)

        # Define training and model checkpoint paths.
        training_dir = command_line_arguments.output_path
        checkpoint_dir = os.path.join(training_dir, "model-checkpoint")
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.logger.info("Created checkpoint directory: %s", checkpoint_dir)

        # Initialize dataset and dataloaders.
        self.logger.info("Retrieving Training and Validation Set for: %s", command_line_arguments.dataset)
        train_loader = dataclass_instance.get_training_data_loader(batch_size=command_line_arguments.batchsize, shuffle_samples=True)
        valid_loader = dataclass_instance.get_validation_data_loader(batch_size=command_line_arguments.batchsize)

        # Create the model.
        self.logger.info("Creating model for training : %s", command_line_arguments.model_type)
        model = create_model(command_line_arguments.model_type, num_targets=dataclass_instance.number_of_classes)
        self.logger.info("Model created with to Finetune for %s-class classifcation problem", dataclass_instance.number_of_classes )

        # Initialize experiment logger.
        self.logger.info("Initializing Experiment Logger")
        experiment_logger = ExperimentLogger(
            output_path=training_dir,
            logger=self.logger
        )
        experiment_logger.display_hyperparamter_for_training(command_line_arguments)

        # Save hyperparameters.
        hyperparameters = vars(command_line_arguments)
        hyperparameters['start_date_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        experiment_logger.save_hyperparameters(hyperparameters)

        # Retrieve optimizer and scheduler based on command-line arguments.
        self.logger.info("Configuring optimizer: %s", command_line_arguments.optimizer)
        optimizer = get_optimizer(command_line_arguments, model)
        self.logger.info("Optimizer configured successfully.")

        self.logger.info("Configuring scheduler: %s", command_line_arguments.scheduler)
        scheduler = get_scheduler(command_line_arguments, optimizer)
        if scheduler is not None:
            self.logger.info("Scheduler configured successfully.")
        else:
            self.logger.info("No scheduler configured.")

        # Create trainer using model's ID to select the appropriate trainer class.
        self.logger.info("Instantiating trainer for model ID: %s", model.model_id)
        trainer = BaseClassificationTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=valid_loader,
            device=device,
            num_epochs=command_line_arguments.epochs,
            checkpoint_dir=training_dir,
            scheduler=scheduler,
            logger=self.logger,
            experiment_logger=experiment_logger,
            multi_label=True
        )
        trainer.run()

    