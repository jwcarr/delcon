from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, ConnectionPatch
from .delaunay_consistency import KeypointReductionResult


def plot_triangulation(axis, keypoints, simplices):
    for simplex in simplices:
        triangle = Polygon(keypoints[simplex], fill=False, edgecolor="Crimson")
        axis.add_patch(triangle)


def plot_connecting_lines(fig, axis1, axis2, keypoints1, keypoints2):
    for point1, point2 in zip(keypoints1, keypoints2):
        connecting_line = ConnectionPatch(
            xyA=point1,
            xyB=point2,
            coordsA="data",
            coordsB="data",
            axesA=axis1,
            axesB=axis2,
            color="MediumSeaGreen",
            linewidth=1,
            linestyle="--",
        )
        fig.add_artist(connecting_line)


def plot_keypoints(axis, keypoints, is_spurious):
    axis.scatter(keypoints[~is_spurious, 0], keypoints[~is_spurious, 1], c="black")
    axis.scatter(keypoints[is_spurious, 0], keypoints[is_spurious, 1], c="blue")


def plot(
    keypoints1: np.ndarray,
    keypoints2: np.ndarray,
    is_spurious: np.ndarray,
    result: KeypointReductionResult,
    output_file_path: Path,
):
    fig, axes = plt.subplots(2, 2, figsize=(20, 20), squeeze=False, sharex=True, sharey=True)

    plot_triangulation(axes[0, 0], keypoints1, result.initial_simplices)
    plot_triangulation(axes[0, 1], keypoints2, result.initial_simplices)

    plot_triangulation(axes[1, 0], result.keypoints1, result.final_simplices)
    plot_triangulation(axes[1, 1], result.keypoints2, result.final_simplices)

    plot_connecting_lines(fig, axes[1, 0], axes[1, 1], result.keypoints1, result.keypoints2)

    plot_keypoints(axes[0, 0], keypoints1, is_spurious)
    plot_keypoints(axes[0, 1], keypoints2, is_spurious)

    plot_keypoints(axes[1, 0], result.keypoints1, result.is_spurious)
    plot_keypoints(axes[1, 1], result.keypoints2, result.is_spurious)

    axes[0, 0].set_xlim(0, 2000)
    axes[0, 0].set_ylim(0, 2000)

    fig.savefig(output_file_path)
