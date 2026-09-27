from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, ConnectionPatch


def plot(
    log: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    output_file_path: Path,
    first_and_last: bool = True,
):
    if first_and_last:
        log = [log[0], log[-1]]
    n_iterations = len(log)
    fig, axes = plt.subplots(n_iterations, 2, figsize=(20, 10 * n_iterations), squeeze=False)
    for row_i, (keypoints1, keypoints2, is_spurious, simplices) in enumerate(log):

        for simplex in simplices:
            triangle = Polygon(keypoints1[simplex], fill=False, edgecolor="Crimson")
            axes[row_i, 0].add_patch(triangle)
        for simplex in simplices:
            triangle = Polygon(keypoints2[simplex], fill=False, edgecolor="Crimson")
            axes[row_i, 1].add_patch(triangle)

        for point1, point2 in zip(keypoints1, keypoints2):
            connecting_line = ConnectionPatch(
                xyA=point1,
                xyB=point2,
                coordsA="data",
                coordsB="data",
                axesA=axes[row_i, 0],
                axesB=axes[row_i, 1],
                color="MediumSeaGreen",
                linewidth=1,
                linestyle="--",
            )
            fig.add_artist(connecting_line)

        axes[row_i, 0].scatter(keypoints1[~is_spurious, 0], keypoints1[~is_spurious, 1], c="black")
        axes[row_i, 1].scatter(keypoints2[~is_spurious, 0], keypoints2[~is_spurious, 1], c="black")

        axes[row_i, 0].scatter(keypoints1[is_spurious, 0], keypoints1[is_spurious, 1], c="blue")
        axes[row_i, 1].scatter(keypoints2[is_spurious, 0], keypoints2[is_spurious, 1], c="blue")

        axes[row_i, 0].set_xlim(0, 2000)
        axes[row_i, 0].set_ylim(0, 2000)
        axes[row_i, 1].set_xlim(0, 2000)
        axes[row_i, 1].set_ylim(0, 2000)

    fig.savefig(output_file_path)
