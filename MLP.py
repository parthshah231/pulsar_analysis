from typing import Any, Optional, Tuple

from numpy import ndarray
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import BCELoss, Linear, ReLU, Sigmoid
from torch.optim import Adam
from torchmetrics.functional import mean_absolute_error


class MLPLightning(LightningModule):
    def __init__(self, in_features, out_features) -> None:
        """
        in_features: Number of input features
        out_features: Number of output features
        """
        super().__init__()
        self.save_hyperparameters()
        self.in_features = in_features
        self.out_features = out_features
        self.bce = BCELoss()
        self.relu = ReLU()
        self.sigmoid = Sigmoid()
        self.linear1 = Linear(self.in_features, 32)
        self.linear2 = Linear(32, self.out_features)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        out = self.sigmoid(x)
        return out

    def training_step(self, batch: Tuple[Tensor, Tensor], *args, **kwargs) -> Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Tuple[Tensor, Tensor], *args, **kwargs) -> Tensor:
        return self._shared_step(batch, "val")

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Tuple[ndarray, ndarray]:
        x = batch
        pred = self(x).squeeze().numpy()
        return pred

    def _shared_step(self, batch: Tuple[Tensor, Tensor], phase: str) -> Tensor:
        # x : [batch_size, in_features]
        # y : [batch_size]
        x, target = batch
        # target: [batch_size] -> [batch_size, 1]
        target = target.unsqueeze(-1)
        pred = self(x)
        loss = self.bce(pred, target)
        self.log(f"{phase}/bce", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = Adam(self.parameters())
        return [opt]
