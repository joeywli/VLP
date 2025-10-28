import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from tqdm import tqdm

from models.mlp import MLP
from models.mlp_online_pico import MLPOnlinePico
from models.pico_interface import PicoInterface

R90_MIN = 10000
R90_MAX = 50000

leds = np.array(
    [
        [22.605263157894736, 17.605263157894736],
        [67.5, 19.0],
        [118.86503067484662, 16.098159509202453],
        [168.00602409638554, 18.99397590361446],
        [216.4935064935065, 17.983766233766232],
        [266.5068493150685, 16.56164383561644],
        [21.42697907188353, 71.89672429481347],
        [67.36892003297609, 63.80956306677659],
        [116.05555555555556, 71.0],
        [166.5, 67.5],
        [211.04736842105265, 58.218947368421055],
        [274.5, 64.5],
        [18.03846153846154, 122.03846153846153],
        [68.23500749625187, 121.25824587706147],
        [119.46458087367178, 121.48347107438016],
        [169.91176470588235, 122.0],
        [218.5, 123.5],
        [267.5302521008403, 118.28151260504201],
        [17.875, 175.17499999999998],
        [69.69359658484525, 172.70021344717182],
        [117.50792393026941, 169.0419968304279],
        [171.5, 170.5],
        [216.47969543147207, 171.03807106598984],
        [271.1503957783641, 170.2678100263852],
        [25.554163845633038, 221.02403520649966],
        [56.96808510638298, 227.98936170212767],
        [117.71626583440425, 224.83761703690104],
        [169.46743447180302, 223.91620333598095],
        [212.7557286892759, 224.65930339138407],
        [268.2879069767442, 223.02093023255816],
        [24.210526315789473, 270.7368421052631],
        [67.4811320754717, 268.5188679245283],
        [124.02187182095626, 269.3565615462869],
        [173.34259259259258, 269.3148148148148],
        [224.48444747612552, 269.7387448840382],
        [268.758064516129, 271.9537634408602],
    ]
)
height = 176.0


def radial_decay_rss(pos, r=0, led_i=0):
    distances = np.linalg.norm(pos - leds[led_i], axis=1)
    rss = np.exp(-distances / r)
    return rss


def calculate_decay_constant(r90_hrs: float):
    decay_k = np.log(1 / 0.9) / r90_hrs
    return decay_k


def is_in_forbidden_area(x, y):
    return (121 <= x < 161) or (121 <= y < 155)


def tksleep(t):
    ms = int(t * 1000)
    root = tk._get_default_root("sleep")
    var = tk.IntVar(root)
    root.after(ms, var.set, 1)
    root.wait_variable(var)


class VLPDemoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Visible Light Positioning under Degradation Demo")

        top_frame = ttk.Frame(root)
        top_frame.pack(side="top", fill="both")

        self.left_frame = ttk.Frame(top_frame)
        self.left_frame.pack(side="left", fill="none", expand=True)

        self.right_frame = ttk.Frame(top_frame)
        self.right_frame.pack(side="right", fill="none", expand=True)

        self.degradation_factors = np.ones(leds.shape[0], dtype=np.float32)
        
        self.baseline_model = None
        self.online_model = None
        self.pico_model = None

        rng = np.random.default_rng(42)
        self.generate_base_rss()
        self.generate_relative_decay(rng)
        self.load_models()
        self.generate_aged_samples(samples_per_timestep=50, rng=rng)

        self.create_led_visualization()
        self.create_positioning_area()
        self.create_widgets()

    def generate_relative_decay(self, rng: np.random.Generator):
        r90_hours = rng.integers(R90_MIN, R90_MAX, 36)
        decay_ks = calculate_decay_constant(r90_hours)

        timesteps = np.arange(0, 100000, 1000)
        relative_decay = np.exp(-np.outer(timesteps, decay_ks), dtype=np.float32)
        relative_decay += rng.normal(0, 0.005, size=relative_decay.shape)

        self.relative_decay = relative_decay
    
    def generate_aged_samples(self, samples_per_timestep: int, rng: np.random.Generator):
        timesteps_n, leds_n = self.relative_decay.shape
        H, W, leds_n = self.data.shape

        # Create a mask for valid data points
        valid_mask = self.data[:, :, 0] != -1

        # Flatten data
        flat_data = self.data.reshape(W*H, leds_n)

        # Generate LED ids
        led_ids = np.arange(leds_n)[None, None, :]  # Shape (1, 1, leds_n)
        # Generate random sample indices at each time step
        valid_flat_idxs = np.flatnonzero(valid_mask)

        sample_flat_idxs = rng.choice(
            valid_flat_idxs, size=(timesteps_n, samples_per_timestep))

        # Fetch the samples for each LED at each time step
        # Add a new axis to the sample_flat_idxs to match the shape of led_ids
        sample_flat_idxs = sample_flat_idxs[:, :, None]
        sample_flat_idxs = np.broadcast_to(
            sample_flat_idxs, (timesteps_n, samples_per_timestep, leds_n))
        led_ids = np.broadcast_to(
            led_ids, (timesteps_n, samples_per_timestep, leds_n))

        # Age the samples
        # Get the samples for each LED at each timestep with the same sample index. Shape (timesteps, samples_per_timestep, leds_n)
        samples = flat_data[sample_flat_idxs, led_ids].clone()
        # Apply the relative decay to the samples
        self.aged_samples = samples * self.relative_decay[:, None, :]

    def generate_base_rss(self):
        xx, yy = np.meshgrid(np.arange(0, 282, 1), np.arange(0, 276, 1))
        matrix_indices = np.array([xx.flatten(), yy.flatten()]).T

        self.base_rss_values = np.zeros((276, 282, 36), dtype=np.float64)
        for i in range(leds.shape[0]):
            self.base_rss_values[:, :, i] = radial_decay_rss(
                matrix_indices, r=30, led_i=i
            ).reshape(276, 282)

    def create_led_visualization(self):
        self.fig = Figure(figsize=(8, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.left_frame)
        self.ax = self.fig.add_subplot()

        img = self.update_led_visualization()
        self.fig.colorbar(img, fraction=0.045, pad=0.04)

        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_led_visualization(self):
        self.ax.clear()
        self.ax.set_title("LED Visualization", fontsize=18)
        self.ax.set_xlabel("x-axis (cm)", fontsize=18)
        self.ax.set_ylabel("y-axis (cm)", fontsize=18)

        rss_values = np.max(self.base_rss_values * self.degradation_factors, axis=2)

        img = self.ax.imshow(rss_values, origin="lower", cmap="viridis", vmin=0, vmax=1)
        self.canvas.draw_idle()

        return img

    def create_positioning_area(self):
        self.fig2 = Figure(figsize=(8, 8), dpi=100)
        # self.fig2.tight_layout(pad=0.5)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.right_frame)
        self.ax2 = self.fig2.add_subplot()

        self.ax2.set_title("Positioning Area", fontsize=18)
        self.ax2.set_xlabel("x-axis (cm)", fontsize=18)
        self.ax2.set_ylabel("y-axis (cm)", fontsize=18)
        self.ax2.grid(True)

        self.ax2.set_xlim(0, 281)
        self.ax2.set_ylim(0, 275)

        # Draw forbidden area
        vertices = [
            (0, 121),
            (121, 121),
            (121, 0),
            (161, 0),
            (161, 121),
            (282, 121),
            (282, 155),
            (161, 155),
            (161, 276),
            (121, 276),
            (121, 155),
            (0, 155),
            (0, 121),
        ]

        forbidden_area = Polygon(
            vertices, closed=True, color="red", alpha=0.1, label="Forbidden Area"
        )
        self.ax2.add_patch(forbidden_area)

        self.gt_pos = plt.Circle(
            (5, 5), radius=3, color="orange", picker=True, label="Ground Truth Position"
        )
        self.ax2.add_patch(self.gt_pos)

        self.predicted_pos = plt.Circle(
            (-1, -1), radius=3, color="C0", picker=True, label="Predicted Position"
        )
        self.ax2.add_patch(self.predicted_pos)

        self.ax2.legend(loc="upper right")
        
        # Set legend font size
        legend = self.ax2.get_legend()
        for text in legend.get_texts():
            text.set_fontsize(16)

        self.dragging = False

        self.canvas2.mpl_connect("button_press_event", self.on_click)
        self.canvas2.mpl_connect("motion_notify_event", self.on_motion)

        self.canvas2.draw()
        self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def load_models(self):
        self.baseline_model = MLP()
        self.baseline_model.load("./saved_runs/MLP-TINY-1748798043.pth")

        self.online_model = MLPOnlinePico(
            data_npy_path="./dataset/heatmaps/heatmap_176/cleaned_LAMBERTIAN-IDW.npy",
            device="cpu",
            seed=42
        )
        self.online_model.load("./saved_runs/MLP-ONLINE-PICO-1750171775.pth")

        self.data = self.online_model.data  # Reuse the data from the model

    def on_click(self, event):
        if event.inaxes != self.ax2:
            return
        contains, _ = self.gt_pos.contains(event)
        if contains and not self.dragging:
            self.dragging = True
            self.canvas2.get_tk_widget().config(cursor="hand2")
        elif self.dragging:
            self.dragging = False
            self.canvas2.get_tk_widget().config(cursor="arrow")

            # Snap the circle to the nearest whole number
            snapped_x = round(event.xdata)
            snapped_y = round(event.ydata)
            self.gt_pos.center = (snapped_x, snapped_y)
            self.canvas2.draw_idle()

    def on_motion(self, event):
        if self.dragging and event.inaxes == self.ax2 and event.xdata and event.ydata:
            self.gt_pos.center = (event.xdata, event.ydata)
            self.canvas2.draw_idle()

            snapped_x = round(event.xdata)
            snapped_y = round(event.ydata)

            if is_in_forbidden_area(snapped_x, snapped_y):
                return

            data_point = self.data[snapped_y, snapped_x, :].reshape(1, -1)

            with torch.no_grad():
                predicted_xy = self.get_model().predict(
                    data_point * self.degradation_factors, eval=True
                )

            self.predicted_pos.center = (
                predicted_xy[0, 0] / 10,
                predicted_xy[0, 1] / 10,
            )
            self.canvas2.draw_idle()
    
    def get_model(self):
        model_type = self.dropdown.get()
        if model_type == "Baseline":
            return self.baseline_model
        elif model_type == "ResidualMLP (Online)":
            return self.online_model
        elif model_type == "Pico":
            if self.pico_model is None:
                self.pico_model = PicoInterface(serial_port="/dev/ttyACM0")
            return self.pico_model

    def create_widgets(self):
        # Create a horizontal frame to hold both button and dropdown
        control_frame = ttk.Frame(self.root)
        control_frame.pack(pady=10)

        # Add button and dropdown side by side
        self.button = ttk.Button(
            control_frame, text="Run Simulation", command=self.run_simulation
        )
        self.button.pack(side="left", padx=5)

        self.dropdown = ttk.Combobox(
            control_frame, values=["Baseline", "ResidualMLP (Online)", "Pico"]
        )
        self.dropdown.pack(side="left", padx=5)

        self.reset_button = ttk.Button(
            control_frame, text="Reset", command=self.reset_simulation
        )
        self.reset_button.pack(side="left", padx=5)

    def send_warmup_batch(self, model_type: str, t: int):
        if model_type == "Baseline":
            # If baseline (without online) is selected, we don't need to
            # do anything special. No learning is involved.
            return
        if model_type == "ResidualMLP (Online)":
            # For online learning, we need to send a warmup batch
            self.online_model.predict(
                self.aged_samples[t, :, :],
                eval=False,
            )
        if model_type == "Pico":
            # For Pico, we need to send a warmup batch
            self.pico_model.predict(
                self.aged_samples[t, :, :],
                eval=False,
            )

    def run_simulation(self):
        model_type = self.dropdown.get()
        if model_type not in ["Baseline", "ResidualMLP (Online)", "Pico"]:
            messagebox.showerror("Error", "Please select a valid model type.")
            return
        self.dropdown.config(state="disabled")

        if model_type == "Pico" and self.pico_model is None:
            self.pico_model = PicoInterface(serial_port="/dev/ttyACM0")

        for t in tqdm(range(100)):
            # Simulate degradation over time
            self.degradation_factors = self.relative_decay[t, :]
            self.update_led_visualization()
            self.send_warmup_batch(model_type, t)
            tksleep(0.05)

        messagebox.showinfo(
            "Aging Complete", "The simulation has finished aging for 100,000 hours."
        )

    def reset_simulation(self):
        
        self.fig.savefig("led_visualization.png")
        self.fig2.savefig("positioning_area.png")
        
        self.degradation_factors = np.ones(leds.shape[0], dtype=np.float32)

        # Make online learning models forget
        self.online_model.scalars = torch.ones(
            self.online_model.data.shape[2],
            dtype=torch.float32,
            device=self.online_model.data.device,
        )

        self.update_led_visualization()
        self.dropdown.config(state="normal")
        messagebox.showinfo("Reset", "The simulation has been reset.")


if __name__ == "__main__":
    root = tk.Tk()
    root.attributes("-fullscreen", True)
    root.attributes('-zoomed', True)
    app = VLPDemoApp(root)
    root.mainloop()
