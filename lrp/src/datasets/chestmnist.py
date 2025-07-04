"""Represents a module containing ChestMNIST dataset."""

import numpy
import torchvision  # type: ignore

from medmnist import ChestMNIST
from lrp.src.datasets.dataset import Dataset, DatasetData


class ChestMnist(Dataset):
    """Represents the classical ChestMNIST dataset."""

    dataset_id = 'chestmnist'
    """Contains a machine-readable ID that uniquely identifies the dataset."""

    def __init__(self, path: str, load_data: bool = True) -> None:
        """Initializes a new Mnist instance.

        Args:
            path (str): The path where the chestMNIST dataset is stored. If it does not exist, it is automatically downloaded to the specified location.
            load_data (bool): If True, download and load the data. If False, only set metadata.
        """

        # Stores the arguments
        self.path = path

        # Exposes some information about the dataset
        self.name = 'ChestMNIST'
        self.path = path
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            #torchvision.transforms.Normalize((0.49361062,), (0.23800726,))
        ])

        # If load_data is True, load the full dataset. Otherwise, skip it.
        if load_data:
            self._load_datasets()
        else:
            # Set empty placeholders if data isn't loaded.
            self._training_data = None
            self._validation_data = None
            self._test_data = None
            self._training_labels = []
            self._validation_labels = []
            self._test_labels = []

    def _load_datasets(self)-> None:
        """Loads all data."""

        # Load training data
        self._training_data = ChestMNIST(
            root=self.path, split='train', download=True, transform=self.transform
        )
        self._training_data.labels = [label[0] for label in self._training_data.labels]
        self._training_labels = self._training_data.labels

        # Load validation data
        self._validation_data = ChestMNIST(
            root=self.path, split='val', download=True, transform=self.transform
        )
        self._validation_data.labels = [label[0] for label in self._validation_data.labels]
        self._validation_labels = self._validation_data.labels

        # Load test data
        self._test_data = ChestMNIST(
            root=self.path, split='test', download=True, transform=self.transform
        )
        self._test_data.labels = [label[0] for label in self._test_data.labels]
        self._test_labels =  self._test_data.labels


    def get_training_labels(self) -> list[int]:
        """Retrieves the labels of the dataset for training.

        Returns:
            list[int]: Returns a list of the labels.
        """

        return self._training_labels

    def get_validation_labels(self) -> list[int]:
        """Retrieves the labels of the dataset for validation.

        Returns:
            list[int]: Returns a list of the labels.
        """

        return self._validation_labels

    def get_test_labels(self) -> list[int]:
        """Retrieves the labels of the dataset for testing.

        Returns:
            list[int]: Returns a list of the labels.
        """

        return self._test_labels

    @property
    def training_data(self) -> DatasetData:
        """Gets the training data of the dataset.

        Returns:
            DatasetData: Returns the training data of the dataset.
        """

        return self._training_data

    @property
    def validation_data(self) -> DatasetData:
        """Gets the validation data of the dataset.

        Returns:
            DatasetData: Returns the validation data of the dataset.
        """

        return self._validation_data

    @property
    def test_data(self) -> DatasetData:
        """Gets the test data of the dataset.

        Returns:
            DatasetData: Returns the validation data of the dataset.
        """

        return self._test_data

    @property
    def sample_shape(self) -> tuple[int, ...]:
        """Gets the the shape of the samples.

        Returns:
            tuple[int, ...]: Returns a tuple that contains the sizes of all dimensions of the samples.
        """
        return tuple([1, 28, 28])

    @property
    def number_of_classes(self) -> int:
        """Gets the number of distinct classes.

        Returns:
            int: Returns the number of distinct classes.
        """
        return len(numpy.unique(self._test_labels))

