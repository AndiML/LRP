"""Represents a module that contains the command for the download of the respective dataset."""

import logging
from argparse import Namespace

from lrp.commands.base import BaseCommand
from lrp.src.datasets.dataset import Dataset


class DownloadDatasetsCommand(BaseCommand):
    """Represents a command that represents the download dataset command."""

    def __init__(self) -> None:
        """Initializes a new DownloadDatasets instance. """

        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)

    def run(self, command_line_arguments: Namespace) -> None:
        """Runs the command.

        Args:
            command_line_arguments (Namespace): The parsed command line arguments.
        """
        
        # Downloads the specified dataset
        self.logger.info("Downloading %s Dataset for OOD Detection", command_line_arguments.dataset.upper(), extra={'start_section': True} )
        Dataset.create(command_line_arguments.dataset, command_line_arguments.dataset_path)

