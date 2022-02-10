from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from torch import rand
from torch.utils.data import Dataset

from constants import TEST_RATIO, VAL_RATIO


class PulsarDataset(Dataset):
    """Prepares custom training and validation dataset for PyTorch Lighning"""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = X
        self.y = y
        self.n_samples = self.X.shape[0]

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.float16]:
        """Enables access to dataset with the help of index"""
        return self.X[index], self.y[index]

    def __len__(self) -> int:
        """Returns the length of dataset"""
        return self.n_samples


class TestDataset(Dataset):
    """Prepares custom testing dataset for PyTorch Lighning"""

    def __init__(self, X: np.ndarray) -> None:
        self.X = X
        self.n_samples = self.X.shape[0]

    def __getitem__(self, index: int) -> Tuple[np.ndarray]:
        return self.X[index]

    def __len__(self) -> int:
        return self.n_samples


def return_datasets(
    df: pd.DataFrame, perform_smote: bool = False
) -> Tuple[Dataset, Dataset, Dataset]:
    """Returns training, validation and testing dataset for MLP model"""
    X = np.array(df.drop(["Classifier"], 1), dtype=np.float32)
    y = np.array(df["Classifier"], dtype=np.float32)

    if perform_smote:
        X_resampled, y_resampled = SMOTE(random_state=43).fit_resample(X, y)
        X_remaining, X_test, y_remaining, y_test = train_test_split(
            X_resampled, y_resampled, test_size=TEST_RATIO
        )
    else:
        X_remaining, X_test, y_remaining, y_test = train_test_split(
            X, y, test_size=TEST_RATIO
        )

    ratio_remaining = 1 - TEST_RATIO
    ratio_val_adjusted = VAL_RATIO / ratio_remaining

    X_train, X_val, y_train, y_val = train_test_split(
        X_remaining, y_remaining, test_size=ratio_val_adjusted
    )

    train_dataset = PulsarDataset(X_train, y_train)
    val_dataset = PulsarDataset(X_val, y_val)
    test_dataset = TestDataset(X_test)

    return train_dataset, val_dataset, test_dataset, y_test
