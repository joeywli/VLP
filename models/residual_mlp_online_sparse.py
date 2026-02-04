import numpy as np
import torch
from ransac_line import fit as fit_ransac_line
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import FPDataset
from models.base import BaseModel

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)

class BottleneckBlock(nn.Module):
    def __init__(self, dim, factor):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, int(dim * factor)),
            nn.Hardswish(),
            nn.Linear(int(dim * factor), dim)
        )
        self.activation = nn.Hardswish()

    def forward(self, x):
        return self.activation(self.net(x) + x)  # residual

class MLPResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = NormalizeInput()

        self.entry = nn.Sequential(
            nn.Linear(9, 256),
            nn.ReLU()
        )

        self.res_block1 = BottleneckBlock(256, 0.1)
        self.res_block2 = BottleneckBlock(256, 0.1)

        self.out = nn.Sequential(
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.norm(x)
        x = self.entry(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        return self.out(x)

class NormalizeInput(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / (x.norm(dim=1, keepdim=True) + 1e-8)

class ResidualMLPOnlineSparse(BaseModel):
    def __init__(self, data_npy_path: str, batch_size=64, lr=0.001, epochs=250, device="cpu", seed=None):
        """
        Initialize the MLP model.

        :param seed: Controls the randomness of the estimator.
        """
        self.model = MLPResNet().to(device)

        self.model.apply(init_weights)

        self.data = torch.tensor(np.load(data_npy_path), dtype=torch.float32, device=device)
        self.scalars = torch.ones(9, dtype=torch.float32, device=device)
        self.min_positions = torch.zeros(2, dtype=torch.long, device=device)
        self.max_positions = torch.tensor([self.data.shape[1] - 1, self.data.shape[0] - 1], dtype=torch.long, device=device)

        self.bin_size = 100

        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.seed = seed

    def fit(self, dataset: FPDataset):
        """
        Fit the model to the dataset.

        :param dataset: The dataset to fit the model to.
        """
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Define loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        torch.manual_seed(self.seed)

        # Training loop
        self.model.train()
        bar = tqdm(range(self.epochs), desc="Training MLP", unit="epoch")
        for _ in bar:
            for X, y in loader:
                optimizer.zero_grad()
                # Use only the 9 selected features for config 2
                outputs = self.model(X[:, [0, 2, 4, 12, 14, 16, 24, 26, 28]])
                # loss = (torch.norm(outputs - y.float(), dim=1)**2).mean()
                loss = criterion(outputs, y.float())
                loss.backward()
                optimizer.step()
            bar.set_postfix({"loss": loss.item()})

    def predict(self, X: torch.Tensor, eval: bool=False) -> torch.Tensor:
        """
        Predict using the model on the dataset.

        :param X: The data to predict on.
        :param eval: Whether or not it's in evaluation mode.
        :return: The predictions.
        """
        self.model.eval()
        
        # Use only the 9 selected features for config 2
        X = X[:, [0, 2, 4, 12, 14, 16, 24, 26, 28]]

        # Apply the scalars to the input data
        X = X * self.scalars
        # Make predictions
        predictions = self.model.forward(X)

        # If in evaluation mode, return the predictions directly
        if eval:
            return predictions
        
        # If in online learning mode, apply RANSAC to the predictions to refine the scalars
        # before returning the predictions

        # Split into bins and apply RANSAC to refine scalars
        X_bins = X.reshape(self.bin_size, -1, X.shape[1])
        predictions_bins = predictions.reshape(self.bin_size, -1, predictions.shape[1])

        # Calculate the positions in the data array
        positions_bins = torch.clamp(torch.round(predictions_bins / 10).long(), min=self.min_positions, max=self.max_positions)

        for i in range(X_bins.shape[1]):
            X_bin = X_bins[:, i, :]
            positions_bin = positions_bins[:, i, :]

            # Extract the references from the data array using the positions
            references = self.data[positions_bin[:, 1], positions_bin[:, 0]][:, [0, 2, 4, 12, 14, 16, 24, 26, 28]]
            # Only consider references that are not -1, which indicates invalid data
            mask = torch.all(references != -1, axis=1)

            # Use RANSAC to refine the scalars
            for i in range(9):
                if mask.sum() < 2: # Need at least 2 points to fit a line
                    continue
                updated_scalar = fit_ransac_line(X_bin[mask, i].cpu().numpy().astype(np.float32), references[mask, i].cpu().numpy().astype(np.float32), threshold=0.1, max_iters=50, seed=self.seed)
                if updated_scalar > 0.5 and updated_scalar < 1.5:
                    self.scalars[i] *= updated_scalar

        return predictions

    def save(self, model_path: str):
        """
        Save the model to the specified path using pickle.

        :param model_path: The path to save the model.
        """
        model_path = model_path + ".pth"
        torch.save(self.model.state_dict(), model_path)
        return model_path

    def load(self, model_path: str):
        """
        Load the model from the specified path using pickle.

        :param model_path: The path to load the model from.
        :return: The loaded model.
        """
        with open(model_path, "rb") as f:
            self.model.load_state_dict(torch.load(f))
            return
        raise ValueError(f"Model at {model_path} not loaded")
