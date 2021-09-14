from model.siren.torchmeta.datasets.triplemnist import TripleMNIST
from model.siren.torchmeta.datasets.doublemnist import DoubleMNIST
from model.siren.torchmeta.datasets.cub import CUB
from model.siren.torchmeta.datasets.cifar100 import CIFARFS, FC100
from model.siren.torchmeta.datasets.miniimagenet import MiniImagenet
from model.siren.torchmeta.datasets.omniglot import Omniglot
from model.siren.torchmeta.datasets.tieredimagenet import TieredImagenet
from model.siren.torchmeta.datasets.tcga import TCGA

from model.siren.torchmeta.datasets import helpers

__all__ = [
    'TCGA',
    'Omniglot',
    'MiniImagenet',
    'TieredImagenet',
    'CIFARFS',
    'FC100',
    'CUB',
    'DoubleMNIST',
    'TripleMNIST',
    'helpers'
]
