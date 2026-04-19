import pandas as pd
import torch
from torch import nn 
from torchinfo import summary
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy, Recall, ConfusionMatrix

# Define the neural network structure
class DefaultModel(nn.Module):
    def __init__(self, input_size):
        super(DefaultModel, self).__init__()
        # Hidden layer
        self.hidden_layer = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=10, bias=True),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        # Output layer 
        self.output_layer = nn.Linear(in_features=10, out_features=1)

    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.output_layer(x)

        return x  
    
def get_training_components(
        model: nn.Module, lr: float 
):
    criterion = nn.BCEWithLogitsLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return criterion, optimizer

# Define the .fit wrapper
class Trainer:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def fit(self, dataloader, epochs):
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for X, y in dataloader:
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(X), y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}: {total_loss:.4f}")


def predict(
        model: nn.Module,
        X: torch.Tensor 
):
    model.eval()

    with torch.no_grad():
        outputs = model(X)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()

        return preds

def evaluate(
        model: nn. Module,
        X: torch.Tensor,
        y: torch.Tensor 
):
    model.eval()

    accuracy = Accuracy(task='binary')
    recall = Recall(task='binary')
    confusion = ConfusionMatrix(task='binary')

    with torch.no_grad():
        outputs = model(X)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.2).float()

        acc = accuracy(preds, y)
        rec = recall(preds, y)
        conf = confusion(preds, y)

    return {
        "accuracy": acc.item(),
        "recall": rec.item(),
        "confusion matrix": conf
    }
    
# Get the dimensions of the df
def get_column(df: pd.DataFrame) -> tuple:
    shape = df.shape[1]
    
    return shape 

# Get a summary how many parameters are going to exist in the nn
def get_summary(
        model: nn.Module, 
        input_size: int,
        col_names: list[str]
):
    return summary(model=model, input_size=input_size,col_names=col_names)

# Transform a df to a torch tensor
def transform_to_torch_tensor(data, is_target=False):
    if isinstance(data, (pd.DataFrame, pd.Series)):
        data = data.values

    tensor = torch.tensor(data, dtype=torch.float32)

    if is_target:
        tensor = tensor.view(-1, 1)

    return tensor

# Transform into a TensorDataset
def transform_to_torch_dataset(X: torch.Tensor, Y: torch.Tensor):
    dataset = TensorDataset(X, Y)

    return dataset 

# Transform a TensorDataset into a DataLoader
def transform_to_dataloader(
        dataset: TensorDataset,
        batch_size: int,
        shuffle = True
):
    
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
