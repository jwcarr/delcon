from pathlib import Path
from .delaunay_consistency import reduce_to_delaunay_consistency
from .synthetic_data import generate_synthetic_keypoints
from .plot import plot


def main() -> None:
    anchor_points, target_points, is_spurious = generate_synthetic_keypoints(
        n_keypoints=40,
        n_spurious=20,
        rotation_deg=10,
        noise_std=10,
        seed=117,
    )
    target_points *= 2
    result = reduce_to_delaunay_consistency(
        anchor_points, target_points, is_spurious, elimination_threshold=0.02
    )
    plot(anchor_points, target_points, is_spurious, result, Path("output.pdf"))
