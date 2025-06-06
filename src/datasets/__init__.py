"""A sub-package that contains datasets and algorithms for partitioning datasets for OOD Detection."""

from lrp.src.datasets.tissuemnist import TissueMnist
from lrp.src.datasets.pathmnist import PathMnist
from lrp.src.datasets.chestmnist import ChestMnist
from lrp.src.datasets.dermamnist import DermaMnist
from lrp.src.datasets.celeba import Celeba


DATASET_IDS = [
    TissueMnist.dataset_id,
    PathMnist.dataset_id,
    ChestMnist.dataset_id,
    DermaMnist.dataset_id,
    Celeba.dataset_id
]

DEFAULT_DATASET_ID = DATASET_IDS[0]
"""Contains the ID of the default dataset."""

__all__ = [
    'TissuemMist',
    'PathMnist',
    'ChestMnist',
    'DermaMnist'

    'DATASET_IDS',
    'DEFAULT_DATASET_ID'
]
