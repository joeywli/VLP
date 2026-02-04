from utils.task_registry import TaskRegistry

from .rf import RF
from .mlp import MLP
from .mlp_online import MLPOnline
from .mlp_online_pico import MLPOnlinePico
from .residual_mlp_online import ResidualMLPOnline
from .residual_mlp_online_sparse import ResidualMLPOnlineSparse

from .knn import KNN
from .wknn import WKNN
from .woknn import WOKNN
from .woknn_online import WOKNNOnline

from .pico_interface import PicoInterface

def register_models() -> TaskRegistry:
    """
    Register all models in the task registry.
    """
    task_registry = TaskRegistry()
    
    # Register models here
    task_registry.register("RF", RF)
    task_registry.register("RF-TINY", RF, n_estimators=6, max_depth=10)
    task_registry.register("MLP", MLP)
    task_registry.register("MLP-TINY", MLP, epochs=25)
    task_registry.register("MLP-TINY-NORMALISE", MLP, epochs=25, normalize=True)
    task_registry.register("MLP-ONLINE-TINY", MLPOnline, data_npy_path="./dataset/heatmaps/heatmap_176/cleaned_LAMBERTIAN-IDW.npy", epochs=25)
    task_registry.register("MLP-ONLINE-PICO", MLPOnlinePico, data_npy_path="./dataset/heatmaps/heatmap_176_augmented_4_downsampled_4/augmented.npy", epochs=50)

    task_registry.register("KNN", KNN, K=5)
    task_registry.register("WKNN", WKNN, K=5)
    task_registry.register("WOKNN", WOKNN, K_range=[10,20])
    task_registry.register("WOKNN-ONLINE", WOKNNOnline, data_npy_path="./dataset/heatmaps/heatmap_176/cleaned_LAMBERTIAN-IDW-BAD_LED.npy", K_range=[10,20])

    task_registry.register("RESIDUAL-MLP-ONLINE", ResidualMLPOnline, data_npy_path="./dataset/heatmaps/heatmap_176/cleaned_LAMBERTIAN-IDW.npy", epochs=50)
    task_registry.register("RESIDUAL-MLP-ONLINE-SPARSE", ResidualMLPOnlineSparse, data_npy_path="./dataset/heatmaps/heatmap_176/cleaned_LAMBERTIAN-IDW.npy", epochs=50)
    task_registry.register("PICO-INTERFACE", PicoInterface, serial_port='COM5')

    return task_registry