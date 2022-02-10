from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pytorch_lightning import Trainer
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from constants import DATA
from dataloader import return_datasets
from MLP import MLPLightning

df = pd.read_csv(DATA / "HTRU_2.csv")
BATCH_SIZE = 10


def test_MLP():
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args(["--gpus=0", "--max_epochs=3", "--val_check_interval=500"])
    train_dataset, val_dataset, test_dataset, targets = return_datasets(
        df, perform_smote=False
    )

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

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        drop_last=False,
    )

    trainer = Trainer.from_argparse_args(args)
    model = MLPLightning(in_features, 1)
    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )
    results = trainer.predict(model, test_dataloader)
    results = np.hstack(results)
    ones = results > 0.5
    acc = (ones == targets).mean()
    matrix = confusion_matrix(targets, ones)
    sns.heatmap(matrix, annot=True)
    plt.show()
    print(acc)


if __name__ == "__main__":
    test_MLP()
