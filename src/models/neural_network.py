import numpy as np
import torch
from torch import nn 
from torch.utils.data import TensorDataset
from pytorch_lightning import Trainer
from ISLP.torch import SimpleDataModule, SimpleModule, ErrorTracker, rec_num_workers
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
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)

        return x
    

# TODOS

# def get_dimension
# def get_summary
# def tranform_to_tensor
# def transform_to_torch_dataset
# defs modules