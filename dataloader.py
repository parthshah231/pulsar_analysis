from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


class PulsarDataset(Dataset):
    def __init__(self) -> None:
        self.df = pd.read_csv(DATA / "HTRU_2.csv")
        self.X = np.array(self.df.drop(["Classifier"], 1), dtype=np.float32)
        self.y = np.array(self.df["Classifier"], dtype=np.float32)
        self.n_samples = self.df.shape[0]

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        return self.X[index], self.y[index]

    def __len__(self) -> int:
        return self.n_samples
