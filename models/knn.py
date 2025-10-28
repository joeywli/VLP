import pickle
import torch
from sklearn.neighbors import KDTree

from dataset import FPDataset
from models.base import BaseModel


class KNN(BaseModel):
    def __init__(self, K=5, device="cpu", seed=None):
        """
        Initialize the KNN model.

        :param seed: Controls the randomness of the estimator.
        """
        self.K = K
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

    def predict(self, X: torch.Tensor, eval: bool=False) -> torch.Tensor:
        """
        Predict using the model on the dataset.

        :param X: The data to predict on.
        :param eval: Whether or not it's in evaluation mode.
        :return: The predictions.
        """
        _, indices = self.kdtree.query(X.numpy(), k=self.K)
        return self.labels[indices].sum(dim=1) / self.K

    def save(self, model_path: str):
        """
        Save the model to the specified path using pickle.

        :param model_path: The path to save the model.
        """
        model_path = model_path + ".kdtree"
        with open(model_path, "wb") as f:
            pickle.dump(self.kdtree, f)
        return model_path

    def load(self, model_path: str):
        """
        Load the model from the specified path using pickle.

        :param model_path: The path to load the model from.
        :return: The loaded model.
        """
        with open(model_path, "rb") as f:
            self.kdtree = pickle.load(f)
            return self.kdtree
        raise ValueError(f"Model at {model_path} not loaded")
