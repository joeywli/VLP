import argparse

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.ndimage import convolve
from scipy.spatial import cKDTree
from tqdm import tqdm
import os

score_threshold = 0.04  # Threshold for score matrix to consider a point bad

def get_tx_positions():
    ##Actual position of the TX (estimated from the LED RSS values)
    return [[22.605263157894736, 17.605263157894736, 176.0], [67.5, 19.0, 176.0], [118.86503067484662, 16.098159509202453, 176.0], [168.00602409638554, 18.99397590361446, 176.0], [216.4935064935065, 17.983766233766232, 176.0], [266.5068493150685, 16.56164383561644, 176.0], [21.42697907188353, 71.89672429481347, 176.0], [67.36892003297609, 63.80956306677659, 176.0], [116.05555555555556, 71.0, 176.0], [166.5, 67.5, 176.0], [211.04736842105265, 58.218947368421055, 176.0], [274.5, 64.5, 176.0], [18.03846153846154, 122.03846153846153, 176.0], [68.23500749625187, 121.25824587706147, 176.0], [119.46458087367178, 121.48347107438016, 176.0], [169.91176470588235, 122.0, 176.0], [218.5, 123.5, 176.0], [267.5302521008403, 118.28151260504201, 176.0], [17.875, 175.17499999999998, 176.0], [69.69359658484525, 172.70021344717182, 176.0], [117.50792393026941, 169.0419968304279, 176.0], [171.5, 170.5, 176.0], [216.47969543147207, 171.03807106598984, 176.0], [271.1503957783641, 170.2678100263852, 176.0], [25.554163845633038, 221.02403520649966, 176.0], [56.96808510638298, 227.98936170212767, 176.0], [117.71626583440425, 224.83761703690104, 176.0], [169.46743447180302, 223.91620333598095, 176.0], [212.7557286892759, 224.65930339138407, 176.0], [268.2879069767442, 223.02093023255816, 176.0], [24.210526315789473, 270.7368421052631, 176.0], [67.4811320754717, 268.5188679245283, 176.0], [124.02187182095626, 269.3565615462869, 176.0], [173.34259259259258, 269.3148148148148, 176.0], [224.48444747612552, 269.7387448840382, 176.0], [268.758064516129, 271.9537634408602, 176.0]]
    # return np.array([[230, 170, 1920], [730, 175, 1920], [1240, 170, 1920], [1740, 170, 1920], [2240, 170, 1920], [2740, 170, 1920],
    #         [230, 670, 1920], [720, 725, 1835], [1230, 670, 1920], [1735, 670, 1920], [2225, 725, 1835], [2725, 670, 1920],
    #         [230, 1170, 1920], [730, 1170, 1920], [1240, 1170, 1920], [1745, 1170, 1920], [2245, 1170, 1920], [2735, 1170, 1920],
    #         [230, 1670, 1920], [730, 1670, 1920], [1240, 1670, 1920], [1745, 1670, 1920], [2245, 1670, 1920], [2735, 1670, 1920],
    #         [230, 2170, 1920], [720, 2225, 1835], [1235, 2170, 1920], [1720, 2170, 1920], [2220, 2225, 1835], [2710, 2170, 1920],
    #         [215, 2670, 1920], [715, 2670, 1920], [1245, 2670, 1920], [1730, 2670, 1920], [2245, 2670, 1920], [2730, 2670, 1920]]) / 10.0

# def get_tx_positions():
#     ##Actual position of the TX (roughly measured)
#     return np.array([[230, 170, 1920], [730, 175, 1920], [1240, 170, 1920], [1740, 170, 1920], [2240, 170, 1920], [2740, 170, 1920],
#             [230, 670, 1920], [720, 725, 1835], [1230, 670, 1920], [1735, 670, 1920], [2225, 725, 1835], [2725, 670, 1920],
#             [230, 1170, 1920], [730, 1170, 1920], [1240, 1170, 1920], [1745, 1170, 1920], [2245, 1170, 1920], [2735, 1170, 1920],
#             [230, 1670, 1920], [730, 1670, 1920], [1240, 1670, 1920], [1745, 1670, 1920], [2245, 1670, 1920], [2735, 1670, 1920],
#             [230, 2170, 1920], [720, 2225, 1835], [1235, 2170, 1920], [1720, 2170, 1920], [2220, 2225, 1835], [2710, 2170, 1920],
#             [215, 2670, 1920], [715, 2670, 1920], [1245, 2670, 1920], [1730, 2670, 1920], [2245, 2670, 1920], [2730, 2670, 1920]]) / 10

def generate_score_matrix(data: npt.NDArray, r=1, bias=0.25) -> npt.NDArray:
    # Create a mask for valid data points
    valid_mask = np.ones_like(data, dtype=float)
    valid_mask[data == -1] = 0

    # Create a kernel for convolution
    kernel_size = 2 * r + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=float)
    kernel[r, r] = 0  # exclude the center

    clipped_data = np.clip(data, 0, None)  # Ensure no negative values for convolution

    # Apply convolution
    refs_sum = convolve(clipped_data, kernel, mode='constant', cval=0.0, axes=(0, 1))
    count_neighbors = convolve(valid_mask, kernel, mode='constant', cval=0.0, axes=(0, 1))

    # We only divide by count_neighbors where it is valid
    refs_mean = np.divide(refs_sum, count_neighbors, where=valid_mask != 0)
    refs_mean[valid_mask == 0] = 0 # Set invalid points to 0

    bias_coef = (1 / (refs_mean + 1e-10)) ** bias

    score_matrix = np.abs(clipped_data - refs_mean) * bias_coef
    return score_matrix

def reconstruct_rss_lambertian(rss_ref, d1, d2, m):
    r"""
    Reconstructs RSS at d1 using known RSS at d2 with Lambertian model.

    rss_ref: RSS value at d2
    d1: Distance from LED to the point to reconstruct (bad point)
    d2: Distance from LED to the reference point (good point)
    m: Lambertian exponent (calculated from the LED positions)

    Small proof, assuming $$ A(d_i) \propto \frac{1}{d_i^2} $$ for i in {1,2}:
    $$
        \frac{I_1^r}{I_2^r} = \frac{A(d_1)}{A(d_2)}\left[\frac{\cos(\phi_1)}{\cos(\phi_2)}\right]^{m+1}
    $$
    $$
        \frac{I_1^r}{I_2^r} = \frac{a \cdot d_1^{-2}}{a \cdot d_2^{-2}}\left[\frac{\frac{h}{d_1}}{\frac{h}{d_2}}\right]^{m+1} = \frac{d_2^2}{d_1^2}\left[\frac{d_2}{d_1}\right]^{m+1} = \left[\frac{d_2}{d_1}\right]^{m+3}
    $$
    $$
        \frac{I_1^r}{I_2^r} = \left[\frac{d_2}{d_1}\right]^{m+3} \implies I_1^r = I_2^r \cdot \left[\frac{d_2}{d_1}\right]^{m+3}
    $$
    """
    
    exponent = m + 3
    return rss_ref * (d2 / d1) ** exponent

strategies = ["MEAN", "IDW", "LAMBERTIAN", "LAMBERTIAN-IDW", "LAMBERTIAN-MEAN"]

def main():
    parser = argparse.ArgumentParser("dataset_clean")
    parser.add_argument(
        "--src", help="Heatmap rank-4 tensor to clean", type=str, default="dataset/heatmaps/heatmap_176/raw.npy"
    )
    parser.add_argument(
        "--dst", help="Folder to export to", type=str, default="dataset/heatmaps/heatmap_176"
    )
    parser.add_argument(
        "--strategy", help="Either LAMBERTIAN or MEAN", type=str, default="MEAN"
    )
    parser.add_argument(
        "--k", help="Number of neighbors to consider", type=int, default=5
    )
    parser.add_argument(
        "--imgs", help="Whether to create images", type=bool, default=False
    )
    args = parser.parse_args()

    if args.strategy not in strategies:
        raise ValueError(f"Invalid strategy: {args.strategy}. Choose from {strategies}.")

    # Load the data
    data = np.load(args.src)

    # Generate the score matrices
    score_matrix = generate_score_matrix(data, r=2, bias=0)
    best_score_idx = score_matrix.argmin(axis=3)  # shape (y, x, led)
    best_scores = np.take_along_axis(score_matrix, best_score_idx[..., np.newaxis], axis=3).squeeze(axis=3) # ...
    best_data = np.take_along_axis(data, best_score_idx[..., np.newaxis], axis=3).squeeze(axis=3) # ...

    if "LAMBERTIAN" in args.strategy:
        leds = get_tx_positions()
        m = - np.log(2) / np.log(np.cos(np.pi / 12))

    h, w, leds_n = best_scores.shape
    
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    coords = np.stack((yy.ravel(), xx.ravel()), axis=-1)
    for i in tqdm(range(leds_n), "Cleaning data for LEDs"):
        # Select the data and score matrix for the current LED
        data_i = best_data[:, :, i]
        best_scores_i = best_scores[:, :, i]

        bad = best_scores_i > score_threshold
        good = ~bad & (data_i != -1) # Good points are those that are not bad and have data

        good_coords = coords[good.ravel()]
        bad_coords = coords[bad.ravel()]

        tree = cKDTree(good_coords)
        dists, idxs = tree.query(bad_coords, k=args.k)

        for j, (y_bad, x_bad) in enumerate(bad_coords):
            d_good = dists[j]
            ixs_good = idxs[j]

            weights = 1 / d_good
            weights /= np.sum(weights)

            if args.strategy == "MEAN":
                # Replace bad point with mean of good points
                good_yx = good_coords[ixs_good]
                rss_values = data_i[good_yx[:, 0], good_yx[:, 1]]
                data_i[y_bad, x_bad] = np.mean(rss_values)
           
            elif args.strategy == "IDW":
                # Replace bad point with IDW of good points
                good_yx = good_coords[ixs_good]
                rss_values = data_i[good_yx[:, 0], good_yx[:, 1]]
                data_i[y_bad, x_bad] = np.sum(rss_values * weights)

            elif args.strategy == "LAMBERTIAN":
                # Replace bad point using Lambertian falloff
                # Get the RSS values of the good points
                good_yx = good_coords[ixs_good]
                rss_values = data_i[good_yx[:, 0], good_yx[:, 1]]

                # Get the RSS value of the first good point (reference)
                rss_ref = rss_values[0]

                # Get the coordinates of the first good point
                y_good, x_good = good_yx[0]

                # Distance from LED to each point
                d1 = np.linalg.norm(leds[i] - np.array([x_bad, y_bad, 0])*10)
                d2 = np.linalg.norm(leds[i] - np.array([x_good, y_good, 0])*10)

                # Replace bad point using Lambertian falloff
                data_i[y_bad, x_bad] = reconstruct_rss_lambertian(rss_ref, d1, d2, m)

            elif args.strategy == "LAMBERTIAN-IDW":
                # Replace bad point using Lambertian falloff
                # Get the RSS values of the good points
                good_yx = good_coords[ixs_good]
                rss_values = data_i[good_yx[:, 0], good_yx[:, 1]]

                # Calculate distances from the LED to the bad point and each good point
                bad_point_3d = np.array([x_bad, y_bad, 0]) # shape: (3,), 10 is the space between points in mm
                good_points_3d = np.column_stack((good_yx[:, 1], good_yx[:, 0], np.zeros(args.k)))  # shape: (k, 3)

                d1 = np.linalg.norm(leds[i] - bad_point_3d)
                d2 = np.linalg.norm(leds[i] - good_points_3d, axis=1) # shape: (k,)

                # Reconstruct RSS values
                reconstructed_rss = reconstruct_rss_lambertian(rss_values, d1, d2, m)  # shape: (k,)
                data_i[y_bad, x_bad] = np.sum(reconstructed_rss * weights)
            elif args.strategy == "LAMBERTIAN-MEAN":
                # Replace bad point using Lambertian falloff
                # Get the RSS values of the good points
                good_yx = good_coords[ixs_good]
                rss_values = data_i[good_yx[:, 0], good_yx[:, 1]]

                # Calculate distances from the LED to the bad point and each good point
                bad_point_3d = np.array([x_bad, y_bad, 0]) # shape: (3,), 10 is the space between points in mm
                good_points_3d = np.column_stack((good_yx[:, 1], good_yx[:, 0], np.zeros(args.k)))  # shape: (k, 3)

                d1 = np.linalg.norm(leds[i] - bad_point_3d)
                d2 = np.linalg.norm(leds[i] - good_points_3d, axis=1) # shape: (k,)

                # Reconstruct RSS values
                reconstructed_rss = reconstruct_rss_lambertian(rss_values, d1, d2, m)  # shape: (k,)
                data_i[y_bad, x_bad] = np.mean(reconstructed_rss)
    
    print(f"Exporting cleaned data to {args.dst}/cleaned_{args.strategy}.npy")

    os.makedirs(args.dst, exist_ok=True)

    np.save(args.dst + f"/cleaned_{args.strategy}.npy", best_data)

    if not args.imgs:
        return

    best_data = np.clip(best_data, 0, None) # Clip negative values to 0
    for i in tqdm(range(leds_n), "Exporting heat maps for cleaned data"):
        plt.imshow(best_data[:, :, i], interpolation='nearest', origin='lower')
        plt.colorbar()
        plt.title(f"Cleaned Data for LED {i} ({args.strategy})")
        plt.savefig(args.dst + f"/led_{i}_cleaned_{args.strategy}.png")
        plt.clf()

if __name__ == "__main__":
    main()