"""Represents a module that contains the descriptor for the command for training and evaluation a model on medical datasets."""

from argparse import ArgumentParser

from lrp.commands.base import BaseCommandDescriptor
from lrp.src.datasets import DATASET_IDS, DEFAULT_DATASET_ID
from lrp.src.models import MODEL_IDS, DEFAULT_MODEL_ID
from lrp.src.training_config import OPTIMIZER_IDS, DEFAULT_OPTIMIZER_ID, SCHEDULER_IDS, DEFAULT_SCHEDULER_ID


class RunLrpCommandDescriptor(BaseCommandDescriptor):
    """Represents the description of the OOD Pipeline command."""

    def get_name(self) -> str:
        """Gets the name of the command.

        Returns:
            str: Returns the name of the command.
        """
        return 'run-lrp'

    def get_description(self) -> str:
        """Gets the description of the command.

        Returns:
            str: Returns the description of the command.
        """
        return 'Runs LRP method with the specified data and model.'

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Adds the command line arguments to the command line argument parser.

        Args:
            parser (ArgumentParser): The command line argument parser to which the arguments are to be added.
        """

        parser.add_argument(
            'output_path',
            type=str,
            help='Path to the directory where experiment results will be saved.'
        )
        parser.add_argument(
            'dataset_path',
            type=str,
            help='Path to the directory where in-distribution and OOD datasets are retrieved or downloaded.'
        )
        parser.add_argument(
            '-d',
            '--dataset',
            type=str,
            default=DEFAULT_DATASET_ID,
            choices=DATASET_IDS,
            help='Name of the dataset to run LRP on.'
        )
        parser.add_argument(
            '-f',
            '--finetune',
            action='store_true',
            help='Wethere to use finetuning before applying LRP.'
        )
        parser.add_argument(
            '-e',
            '--epochs',
            type=int,
            default=5,
            help="Number of training epochs."
        )
        parser.add_argument(
            '-b',
            '--batchsize',
            type=int,
            default=64,
            help="Batch size during training."
        )
        parser.add_argument(
            '-M',
            '--multiclass',
            action='store_true',
            help="Use multiclass training."
        )
        parser.add_argument(
            '-l',
            '--learning_rate',
            type=float,
            default=0.001,
            help='Learning rate used during training.'
        )

        parser.add_argument(
            '-m',
            '--momentum',
            type=float,
            default=0.9,
            help='Momentum for the optimizer.'
        )

        parser.add_argument(
            '-w',
            '--weight_decay',
            type=float,
            default=0.0005,
            help='Weight decay used in the optimizer.'
        )

        # Model arguments
        parser.add_argument(
            '-t',
            '--model_type',
            type=str,
            default=DEFAULT_MODEL_ID,
            choices=MODEL_IDS,
            help='Type of neural network architecture used for finetuning.'
        )
        parser.add_argument(
            '-g',
            '--use_gpu',
            action='store_true',
            help="If set, CUDA is utilized for training."
        )
        # Optimizer arguments
        parser.add_argument(
            '-p',
            '--optimizer',
            type=str,
            default=DEFAULT_OPTIMIZER_ID,
            choices=OPTIMIZER_IDS,
            help="Type of optimizer to use."
        )

        # Scheduler arguments
        parser.add_argument(
            '-s',
            '--scheduler',
            type=str,
            default=DEFAULT_SCHEDULER_ID,
            choices=SCHEDULER_IDS,
            help="Type of learning rate scheduler to use."
        )

        # Scheduler-specific arguments
        parser.add_argument(
            '-S',
            '--step_size',
            type=int,
            default=10,
            help='Step size for StepLR scheduler (default: 10).'
        )

        parser.add_argument(
            '-G',
            '--gamma',
            type=float,
            default=0.1,
            help='Decay factor for StepLR or ExponentialLR scheduler (default: 0.1).'
        )

        parser.add_argument(
            '-F',
            '--learning_rate_factor',
            type=float,
            default=0.1,
            help='Factor by which the LR is reduced in ReduceLROnPlateau scheduler (default: 0.1).'
        )

        parser.add_argument(
            '-A',
            '--learning_rate_patience',
            type=int,
            default=5,
            help='Number of epochs with no improvement before reducing LR in ReduceLROnPlateau scheduler (default: 5).'
        )

        parser.add_argument(
            '-N',
            '--num_iteration_max',
            type=int,
            default=50,
            help='Maximum iterations for CosineAnnealingLR scheduler (default: 50).'
        )

        parser.add_argument(
            '-R',
            '--minimum_learning_rate',
            type=float,
            default=0.0,
            help='Minimum learning rate for cosine annealing schedulers (default: 0.0).'
        )

        parser.add_argument(
            '-I',
            '--learning_increase_restart',
            type=int,
            default=2,
            help='Factor for increasing the restart period in CosineAnnealingWarmRestarts scheduler (default: 2).'
        )

        parser.add_argument(
            '-C',
            '--num_iteration_restart',
            type=int,
            default=10,
            help='Number of iterations for a restart in CosineAnnealingWarmRestarts scheduler (default: 10).'
        )