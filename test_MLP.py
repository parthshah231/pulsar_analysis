from argparse import ArgumentParser

import numpy as np
import pandas as pd
from pytorch_lightning import Trainer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from constants import DATA, TEST_RATIO, VAL_RATIO
from dataloader import PulsarDataset
from MLP import MLPLightning

df = pd.read_csv(DATA / "HTRU_2.csv")
X = np.array(df.drop(["Classifier"], 1), dtype=np.float32)
y = np.array(df["Classifier"], dtype=np.float32)

X_remaining, X_test, y_remaining, y_test = train_test_split(X, y, test_size=TEST_RATIO)

ratio_remaining = 1 - TEST_RATIO
ratio_val_adjusted = VAL_RATIO / ratio_remaining

X_train, X_val, y_train, y_val = train_test_split(
    X_remaining, y_remaining, test_size=ratio_val_adjusted
)

BATCH_SIZE = 10


def test_MLP():
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args(["--gpus=0", "--max_epochs=2", "--val_check_interval=500"])
    train_dataset = PulsarDataset(X_train, y_train)
    val_dataset = PulsarDataset(X_val, y_val)

    in_features = len(train_dataset[0][0])
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        drop_last=False,
    )

    trainer = Trainer.from_argparse_args(args)
    model = MLPLightning(in_features, 1)
    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )


if __name__ == "__main__":
    test_MLP()
