from pathlib import Path
from .delaunay_consistency import reduce_to_delaunay_consistency
from .synthetic_data import generate_synthetic_keypoints
from .plot import plot


def main() -> None:
    anchor_points, target_points, is_spurious = generate_synthetic_keypoints(
        n_keypoints=400,
        n_spurious=100,
        rotation_deg=10,
        noise_std=10,
        seed=117,
    )
    target_points *= 2
    anchor_points_prime, target_points_prime, log = reduce_to_delaunay_consistency(
        anchor_points, target_points, is_spurious, elimination_threshold=0.1
    )
    plot(log, Path("output.pdf"))
