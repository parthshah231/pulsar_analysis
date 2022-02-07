from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from constants import DATA


class PulsarDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = X
        self.y = y
        self.n_samples = self.X.shape[0]

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.float16]:
        return self.X[index], self.y[index]

    def __len__(self) -> int:
        return self.n_samples


class TestDataset(Dataset):
    def __init__(self, X: np.ndarray) -> None:
        self.X = X
        self.n_samples = self.X.shape[0]

    def __getitem__(self, index: int) -> Tuple[np.ndarray]:
        return self.X[index]

    def __len__(self) -> int:
        return self.n_samples
