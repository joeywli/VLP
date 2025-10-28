import pickle
from matplotlib import pyplot as plt
import numpy as np
import torch
from sklearn.neighbors import KDTree
from tqdm import tqdm
import torch.nn as nn

from ransac_line import fit as fit_ransac_line

from dataset import FPDataset
from models.base import BaseModel

class OptimalKClassifier(nn.Module):
    def __init__(self, num_k_classes, hidden_dim=5, device="cpu"):
        """
        Small neural network to classify optimal K from LED input.
        
        Args:
            num_k_classes (int): Number of possible K values (e.g., 5 for K in [2,3,4,5,6])
            hidden_dim (int): Size of hidden layers
        """
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(36, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_k_classes)
        )
    
    def forward(self, x):
        """
        Forward pass.
        Args:
            x (Tensor): shape (batch_size, 36)
        Returns:
            logits (Tensor): shape (batch_size, num_k_classes)
        """
        return self.net(x)

class WOKNNOnline(BaseModel):
    def __init__(self, data_npy_path: str, K_range=[2, 7], device="cpu", seed=None):
        """
        Initialize the KNN model.

        :param seed: Controls the randomness of the estimator.
        """
        self.K_range = K_range
        self.opt_k_model = OptimalKClassifier(num_k_classes=K_range[1]-K_range[0]+1, device=device).to(device)
        
        self.data = torch.tensor(np.load(data_npy_path), dtype=torch.float32, device=device)
        self.scalars = torch.ones(self.data.shape[2], dtype=torch.float32, device=device)
        self.min_positions = torch.zeros(2, dtype=torch.long, device=device)
        self.max_positions = torch.tensor([self.data.shape[1] - 1, self.data.shape[0] - 1], dtype=torch.long, device=device)
        
        self.device = device
        self.seed = seed

        if self.device != "cpu":
            raise ValueError("KNN only supports CPU mode.")

    def fit(self, dataset: FPDataset):
        """
        Fit the model to the dataset.

        :param dataset: The dataset to fit the model to.
        """
        # Fit distance tree to dataset
        self.kdtree = KDTree(dataset.leds.numpy())
        self.labels = dataset.xy

        # Find optimal K using leave-one-out cross-validation
        errors = np.zeros(
            (dataset.xy.shape[0], self.K_range[1] - self.K_range[0] + 1))
        k_bar = tqdm(range(self.K_range[0], self.K_range[1]+1))
        for K in k_bar:
            k_bar.set_description(
                f"Finding optimal K for all points, current K={K}")

            distances, indices = self.kdtree.query(dataset.leds.numpy(), k=K+1)
            # Exclude the point itself
            distances, indices = distances[:, 1:], indices[:, 1:]
            predicted_labels = dataset.xy[indices].sum(axis=1) / K
            errors[:, K - self.K_range[0]
                   ] = (dataset.xy - predicted_labels).norm(dim=1).numpy()

        best_Ks = errors.argmin(axis=1) + self.K_range[0]
        # plot_mean_errors_k(errors.mean(axis=0), self.K_range) # UNCOMMENT IF YOU WANT TO SEE MEAN ERRORS PER K
        # plot_best_ks_histo(best_Ks) # UNCOMMENT IF YOU WANT TO SEE A HISTOGRAM OF BEST KS
        plot_best_ks(dataset.xy, best_Ks) # UNCOMMENT IF YOU WANT TO SEE KS

        # Train optimal K classifier
        optimizer = torch.optim.Adam(self.opt_k_model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        num_epochs = 50
        batch_size = 128
        
        # Train loop
        bar = tqdm(range(num_epochs), desc="Training optimal K classifier | loss: N/A")
        for _ in bar:
            permutation = torch.randperm(dataset.leds.size(0))
            for i in range(0, dataset.leds.size(0), batch_size):
                indices = permutation[i:i+batch_size]
                batch_leds = dataset.leds[indices].to(self.device)
                batch_best_Ks = torch.tensor(
                    best_Ks[indices.numpy()] - self.K_range[0], dtype=torch.long).to(self.device)

                optimizer.zero_grad()
                outputs = self.opt_k_model(batch_leds)
                loss = criterion(outputs, batch_best_Ks)
                bar.set_description(f"Training optimal K classifier | loss: {loss.item():.4f}")
                loss.backward()
                optimizer.step()

    def predict(self, X: torch.Tensor, eval: bool = False) -> torch.Tensor:
        """
        Predict using the model on the dataset.

        :param X: The data to predict on.
        :param eval: Whether or not it's in evaluation mode.
        :return: The predictions.
        """
        
        self.opt_k_model.eval()
        
        X = X * self.scalars
        
        with torch.no_grad():
            K = self.opt_k_model(X.to(self.device)).argmax(dim=1).cpu().numpy() + self.K_range[0]
        distances, indices = self.kdtree.query(X.numpy(), k=self.K_range[1])
        
        # For all inputs, select the appropriate number of neighbors based on predicted K
        predictions = torch.zeros((X.shape[0], 2))
        for i in range(X.shape[0]):
            k_i = K[i]
            dists_i = distances[i, :k_i]
            inds_i = indices[i, :k_i]
            
            total_distance = (1 / dists_i).sum()
            weights = (1 / dists_i) / (total_distance + 1e-8)
            
            predictions[i] = (self.labels[inds_i] * weights[:, None]).sum(
                axis=0) / weights.sum()
        
        if eval:
            return predictions
        
        positions = torch.clamp(torch.round(predictions / 10).long(), min=self.min_positions, max=self.max_positions)
        
        # Extract the references from the data array using the positions
        references = self.data[positions[:, 1], positions[:, 0]]
        # Only consider references that are not -1, which indicates invalid data
        mask = torch.all(references != -1, axis=1)

        # Use RANSAC to refine the scalars
        for i in range(36):
            self.scalars[i] *= fit_ransac_line(X[mask, i].cpu().numpy().astype(np.float32), references[mask, i].cpu().numpy().astype(np.float32), threshold=1.0, max_iters=10, seed=self.seed)

        return predictions

    def save(self, model_path: str):
        """
        Save the model to the specified path using pickle.

        :param model_path: The path to save the model.
        """
        with open(model_path  + ".kdtree", "wb") as f:
            pickle.dump(self.kdtree, f)
        torch.save(self.opt_k_model.state_dict(), model_path + "-optkmodel.pt")
        torch.save(self.labels, model_path + "-labels.pt")
        return model_path

    def load(self, model_path: str):
        """
        Load the model from the specified path using pickle.

        :param model_path: The path to load the model from.
        :return: The loaded model.
        """
        with open(model_path + ".kdtree", "rb") as f:
            self.kdtree = pickle.load(f)
            with open(model_path + "-optkmodel.pt", "rb") as f:
                self.opt_k_model.load_state_dict(torch.load(f))
                with open(model_path + "-labels.pt", "rb") as f:
                    self.labels = torch.load(f)
                    return
        raise ValueError(f"Model at {model_path} not loaded")


def plot_mean_errors_k(mean_errors, k_range):
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(k_range[0], k_range[1]+1), mean_errors, marker='o')
    plt.xlabel('K')
    plt.ylabel('Mean Error')
    plt.title('Mean Errors for Different K Values')
    plt.grid()
    plt.show()

def plot_best_ks_histo(best_Ks):
    plt.figure(figsize=(8, 4))
    plt.hist(best_Ks, bins=np.arange(best_Ks.min(), best_Ks.max()+2)-0.5, edgecolor='black')
    plt.xlabel('Optimal K')
    plt.ylabel('Frequency')
    plt.title('Histogram of Optimal K Values')
    plt.grid()
    plt.show()

def plot_best_ks(labels, best_Ks, grid_size=200):
    x = labels[:, 0]
    y = labels[:, 1]

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    grid_x = np.linspace(x_min, x_max, grid_size)
    grid_y = np.linspace(y_min, y_max, grid_size)
    grid = np.full((grid_size, grid_size), np.nan)

    xi = np.searchsorted(grid_x, x) - 1
    yi = np.searchsorted(grid_y, y) - 1

    valid = (xi >= 0) & (xi < grid_size) & (yi >= 0) & (yi < grid_size)
    xi, yi = xi[valid], yi[valid]
    best_Ks = best_Ks[valid]

    grid[yi, xi] = best_Ks

    plt.figure(figsize=(6, 6))
    im = plt.imshow(
        grid,
        extent=(x_min, x_max, y_min, y_max),
        origin='lower',
        cmap='viridis',
        interpolation='nearest',
        aspect='auto'
    )
    plt.colorbar(im, label='Optimal K')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('Optimal K Across XY Space')
    plt.show()
