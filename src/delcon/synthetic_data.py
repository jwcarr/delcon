import numpy as np


def generate_synthetic_keypoints(
    n_keypoints,
    n_spurious=0,
    offset=(0.0, 0.0),
    rotation_deg=0.0,
    noise_std=0.0,
    seed=None,
):
    """
    Generate two sets of synthetic corresponding 2D keypoints.

    points_a contains the original random keypoints.

    points_b contains corresponding points transformed by:
        1. Rotation
        2. Translation
        3. Gaussian noise

    A configurable number of correspondences can then be made
    spurious by replacing their points in points_b with independently
    generated random points.

    Parameters
    ----------
    n_keypoints : int
        Total number of keypoints/correspondences.

    n_spurious : int
        Number of deliberately incorrect correspondences.

    offset : tuple of float
        Translation (dx, dy) applied to the second set.

    rotation_deg : float
        Rotation angle in degrees, applied around the center
        of the first point cloud.

    noise_std : float
        Standard deviation of Gaussian noise added independently
        to each x/y coordinate of the second set.

    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    points_a : ndarray, shape (N, 2)
        Original keypoints.

    points_b : ndarray, shape (N, 2)
        Transformed/noisy keypoints, with n_spurious deliberately
        incorrect correspondences.

    spurious_indices : ndarray
        Indices of the deliberately spurious correspondences.
    """

    if n_keypoints < 1:
        raise ValueError("n_keypoints must be at least 1")

    if not 0 <= n_spurious <= n_keypoints:
        raise ValueError("n_spurious must be between 0 and n_keypoints")

    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    width = 1000
    height = 1000

    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------
    # 1. Generate the original points
    # ---------------------------------------------------------

    points_a = rng.uniform(
        low=[0.0, 0.0],
        high=[width - 200, height - 200],
        size=(n_keypoints, 2),
    )
    points_a += 100

    # ---------------------------------------------------------
    # 2. Create the corresponding points
    # ---------------------------------------------------------

    center = points_a.mean(axis=0)

    theta = np.deg2rad(rotation_deg)

    rotation_matrix = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )

    # Rotate around the center of the point cloud
    points_b = (points_a - center) @ rotation_matrix.T + center

    # Apply translation
    points_b += np.asarray(offset)

    # Add general Gaussian noise
    points_b += rng.normal(
        loc=0.0,
        scale=noise_std,
        size=points_b.shape,
    )

    # ---------------------------------------------------------
    # 3. Replace selected correspondences with random points
    # ---------------------------------------------------------

    if n_spurious > 0:

        spurious_indices = rng.choice(
            n_keypoints,
            size=n_spurious,
            replace=False,
        )

        points_b[spurious_indices] = rng.uniform(
            low=[0.0, 0.0],
            high=[width, height],
            size=(n_spurious, 2),
        )

    else:
        spurious_indices = np.array([], dtype=int)

    is_spurious = np.zeros(n_keypoints, dtype=bool)
    is_spurious[spurious_indices] = True

    return points_a, points_b, is_spurious
