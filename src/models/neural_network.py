import numpy as np
import pandas as pd
import torch
from torch import nn 
from torchinfo import summary
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torchmetrics import Accuracy

class DefaultModel(nn.Module):
    def __init__(self, input_size):
        super(DefaultModel, self).__init__()
        # Hidden layer
        self.hidden_layer = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=10, bias=True),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        # Output layer 
        self.output_layer = nn.Linear(in_features=10, out_features=1)

    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)

        return x

class RegressionModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)

        loss = self.loss_fn(preds, y)
        
        probs = torch.sigmoid(preds)
        acc = self.acc(probs, y.int())

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)

        self.log("val_loss", loss)
        self.log("val_mae", self.mae(preds, y))

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)

        self.log("test_loss", loss)
        self.log("test_mae", self.mae(preds, y))

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(), lr=1e-3)
    
    

def get_dimension(df: pd.DataFrame) -> tuple:
    shape = df.shape[1]
    
    return shape 

def get_summary(
        model: nn.Module, 
        input_size: int,
        col_names: list[str]
):
    return summary(model=model, input_size=input_size,col_names=col_names)

def transform_to_torch_tensor(df: pd.DataFrame):
    tensor = torch.tensor(df.astype(np.float32))

    return tensor 

def transform_to_torch_dataset(X: torch.Tensor, Y: torch.Tensor):
    dataset = TensorDataset(X, Y)

    return dataset 

def get_DataLoader(
        dataset: TensorDataset,
        batch_size: int,
        shuffle = True
):
    
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

# defs modules