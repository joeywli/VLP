import argparse
import os

import numpy as np
import numpy.typing as npt

from tqdm import tqdm

def get_tx_positions():
    ##Actual position of the TX (roughly measured)
    return np.array([[230, 170, 1920], [730, 175, 1920], [1240, 170, 1920], [1740, 170, 1920], [2240, 170, 1920], [2740, 170, 1920],
            [230, 670, 1920], [720, 725, 1835], [1230, 670, 1920], [1735, 670, 1920], [2225, 725, 1835], [2725, 670, 1920],
            [230, 1170, 1920], [730, 1170, 1920], [1240, 1170, 1920], [1745, 1170, 1920], [2245, 1170, 1920], [2735, 1170, 1920],
            [230, 1670, 1920], [730, 1670, 1920], [1240, 1670, 1920], [1745, 1670, 1920], [2245, 1670, 1920], [2735, 1670, 1920],
            [230, 2170, 1920], [720, 2225, 1835], [1235, 2170, 1920], [1720, 2170, 1920], [2220, 2225, 1835], [2710, 2170, 1920],
            [215, 2670, 1920], [715, 2670, 1920], [1245, 2670, 1920], [1730, 2670, 1920], [2245, 2670, 1920], [2730, 2670, 1920]])/10


leds = get_tx_positions()

# leds = np.array([[230, 170, 1920], [730, 175, 1920], [1240, 170, 1920], [1740, 170, 1920], [2240, 170, 1920], [2740, 170, 1920],
#             [230, 670, 1920], [720, 725, 1835], [1230, 670, 1920], [1735, 670, 1920], [2225, 725, 1835], [2725, 670, 1920],
#             [230, 1170, 1920], [730, 1170, 1920], [1240, 1170, 1920], [1745, 1170, 1920], [2245, 1170, 1920], [2735, 1170, 1920],
#             [230, 1670, 1920], [730, 1670, 1920], [1240, 1670, 1920], [1745, 1670, 1920], [2245, 1670, 1920], [2735, 1670, 1920],
#             [230, 2170, 1920], [720, 2225, 1835], [1235, 2170, 1920], [1720, 2170, 1920], [2220, 2225, 1835], [2710, 2170, 1920],
#             [215, 2670, 1920], [715, 2670, 1920], [1245, 2670, 1920], [1730, 2670, 1920], [2245, 2670, 1920], [2730, 2670, 1920]]) / 10.0

def reconstruct_rss_lambertian(rss_ref, d1, d2, m = 19.9937273585):
    r"""
    Reconstructs RSS at d1 using known RSS at d2 with Lambertian model.

    rss_ref: RSS value at d2
    d1: Distance from LED to the point to reconstruct (bad point)
    d2: Distance from LED to the reference point (good point)
    m: Lambertian exponent (calculated from the LED positions)

    Check clean.py for the proof of this formula.
    """

    exponent = m + 3
    return rss_ref * (d2 / d1) ** exponent

def upsample_to_grid(data: npt.NDArray[np.float64], factor: int, fill_value=-1) -> npt.NDArray[np.float64]:
    grid = np.full((data.shape[0]*factor, data.shape[1]*factor, data.shape[2]), fill_value, dtype=data.dtype)
    grid[::factor, ::factor] = data
    return grid

def augment_data(data: npt.NDArray[np.float64], fill_value=-1, q: int = 1, offset_x: int = 0, offset_y: int = 0) -> npt.NDArray[np.float64]:
    y_indices, x_indices, l_indices = np.where(data == fill_value)

    for y, x, no_led in tqdm(zip(y_indices, x_indices, l_indices), total=len(y_indices), desc=f"Augmenting data q={q}/4"):
        amt = 0

        data[y, x, no_led] = 0

        nearest_y = np.clip(int(np.ceil(y / 4) * 4), 0, data.shape[0] - 1)
        nearest_x = np.clip(int(np.ceil(x / 4) * 4), 0, data.shape[1] - 1)

        d1 = np.linalg.norm(leds[no_led] + np.array([offset_x, offset_y, 0]) - np.array([x, y, 0]))
        d2 = np.linalg.norm(leds[no_led] + np.array([offset_x, offset_y, 0]) - np.array([nearest_x, nearest_y, 0]))

        reconstructed_rss = reconstruct_rss_lambertian(data[nearest_y, nearest_x, no_led], d1, d2)
        if reconstructed_rss > 0:
            data[y, x, no_led] += reconstructed_rss
            amt += 1

        nearest_y = np.clip(int(np.floor(y / 4) * 4), 0, data.shape[0] - 1)
        nearest_x = np.clip(int(np.floor(x / 4) * 4), 0, data.shape[1] - 1)

        d1 = np.linalg.norm(leds[no_led] + np.array([offset_x, offset_y, 0]) - np.array([x, y, 0]))
        d2 = np.linalg.norm(leds[no_led] + np.array([offset_x, offset_y, 0]) - np.array([nearest_x, nearest_y, 0]))

        reconstructed_rss = reconstruct_rss_lambertian(data[nearest_y, nearest_x, no_led], d1, d2)
        if reconstructed_rss > 0:
            data[y, x, no_led] += reconstructed_rss
            amt += 1

        nearest_y = np.clip(int(np.ceil(y / 4) * 4), 0, data.shape[0] - 1)
        nearest_x = np.clip(int(np.floor(x / 4) * 4), 0, data.shape[1] - 1)

        d1 = np.linalg.norm(leds[no_led] + np.array([offset_x, offset_y, 0]) - np.array([x, y, 0]))
        d2 = np.linalg.norm(leds[no_led] + np.array([offset_x, offset_y, 0]) - np.array([nearest_x, nearest_y, 0]))

        reconstructed_rss = reconstruct_rss_lambertian(data[nearest_y, nearest_x, no_led], d1, d2)
        if reconstructed_rss > 0:
            data[y, x, no_led] += reconstructed_rss
            amt += 1

        nearest_y = np.clip(int(np.floor(y / 4) * 4), 0, data.shape[0] - 1)
        nearest_x = np.clip(int(np.ceil(x / 4) * 4), 0, data.shape[1] - 1)

        d1 = np.linalg.norm(leds[no_led] + np.array([offset_x, offset_y, 0]) - np.array([x, y, 0]))
        d2 = np.linalg.norm(leds[no_led] + np.array([offset_x, offset_y, 0]) - np.array([nearest_x, nearest_y, 0]))

        reconstructed_rss = reconstruct_rss_lambertian(data[nearest_y, nearest_x, no_led], d1, d2)
        if reconstructed_rss > 0:
            data[y, x, no_led] += reconstructed_rss
            amt += 1
        
        data[y, x, no_led] /= 4

    return data

def main():
    parser = argparse.ArgumentParser("dataset_downsample")
    parser.add_argument(
        "--src", help="Folder to import from", type=str, default="dataset/heatmaps/heatmap_176_downsampled_4"
    )
    parser.add_argument(
        "--dst", help="File to export to", type=str, default="dataset/heatmaps/heatmap_176_augmented_4_downsampled_4/augmented.npy"
    )
    parser.add_argument(
        "--factor", help="Augmenting factor", type=int, default=4
    )
    parser.add_argument(
        "--cross_x_start", help="Cross x start", type=int, default=121
    )
    parser.add_argument(
        "--cross_x_end", help="Cross x end", type=int, default=161
    )
    parser.add_argument(
        "--cross_y_start", help="Cross y start", type=int, default=121
    )
    parser.add_argument(
        "--cross_y_end", help="Cross y end", type=int, default=155
    )
    parser.add_argument(
        "--final_size_x", help="Final size x", type=int, default=282
    )
    parser.add_argument(
        "--final_size_y", help="Final size y", type=int, default=276
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.dst), exist_ok=True)

    print(f"Loading data from {args.src}")
    q1 = np.load(f"{args.src}/q1_downsampled.npy")
    q2 = np.load(f"{args.src}/q2_downsampled.npy")
    q3 = np.load(f"{args.src}/q3_downsampled.npy")
    q4 = np.load(f"{args.src}/q4_downsampled.npy")

    upsampled_q1 = upsample_to_grid(q1, args.factor)
    upsampled_q2 = upsample_to_grid(q2, args.factor)
    upsampled_q3 = upsample_to_grid(q3, args.factor)
    upsampled_q4 = upsample_to_grid(q4, args.factor)

    upsampled_q1 = augment_data(upsampled_q1, q=1, offset_x=args.cross_x_end, offset_y=args.cross_y_end)
    upsampled_q2 = augment_data(upsampled_q2, q=2, offset_x=0, offset_y=args.cross_y_end)
    upsampled_q3 = augment_data(upsampled_q3, q=3, offset_x=0, offset_y=0)
    upsampled_q4 = augment_data(upsampled_q4, q=4, offset_x=args.cross_x_end, offset_y=0)

    upsampled = np.full((args.final_size_y, args.final_size_x, len(leds)), -1, dtype=np.float64)
    upsampled[args.cross_y_end:, args.cross_x_end:] = upsampled_q1[:args.final_size_y-args.cross_y_end, :args.final_size_x-args.cross_x_end]
    upsampled[args.cross_y_end:, :args.cross_x_start] = upsampled_q2[:args.final_size_y-args.cross_y_end, :args.cross_x_start]
    upsampled[:args.cross_y_start, :args.cross_x_start] = upsampled_q3[:args.cross_y_start, :args.cross_x_start]
    upsampled[:args.cross_y_start, args.cross_x_end:] = upsampled_q4[:args.cross_y_start, :args.final_size_x-args.cross_x_end]

    np.save(args.dst, upsampled)
    print(f"Exported data to {args.dst}")

if __name__ == "__main__":
    main()