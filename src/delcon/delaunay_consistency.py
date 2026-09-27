from itertools import combinations
import numpy as np
from scipy.spatial import Delaunay


def calculate_equilateralness(triangle: np.ndarray) -> float:
    """
    Given a triangle (array of shape (3, 2)), returns an "equilateralness"
    score in (0, 1], where 1 is equilateral and 0 is degenerate. The score
    is essentially the ratio of the triangle's area to the area of an
    equilateral triangle with the same perimeter.
    """
    assert len(triangle) == 3
    side_lengths = [
        np.linalg.norm(vertex1 - vertex2) for vertex1, vertex2 in combinations(triangle, 2)
    ]
    perimeter = sum(side_lengths)
    semiperimeter = perimeter / 2.0
    area = np.sqrt(
        semiperimeter * np.prod([semiperimeter - side_length for side_length in side_lengths])
    )
    area_of_an_equilateral_triangle_of_identical_perimeter = perimeter**2 / (12 * np.sqrt(3))
    equilateralness = area / area_of_an_equilateral_triangle_of_identical_perimeter
    if equilateralness <= 0:
        return 0.0000000000001
    return equilateralness


def reduce_to_delaunay_consistency(
    keypoints1: np.ndarray,
    keypoints2: np.ndarray,
    is_spurious: np.ndarray,
    elimination_threshold: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Given two sets of matched keypoints, removes the keypoint pairs that do
    not satisfy Delaunay consistency. For a keypoint pair to satisfy Delaunay
    consistency the triangles incident to a keypoint in image 1 (in a
    Delaunay triangulation) should retain their shape when imposed into image
    2. I.e., the triangular shape that exists between three landmarks in one
    image should remain the same when imposed onto the other image, even
    under changes in rotation, scale, and translation.

    The algorithm constructs a Delaunay triangulation from the first set
    of keypoints and imposes that triangulation onto the second set of
    keypoints. For each triangle, we compute its "equilateralness" and the
    equilateralness of the equivalent triangle in the second image. If the
    two equilateralness scores are similar, the vertices of the triangle are
    likely to be good keypoint matches. If the triangles are dissimilar, one
    or more of its vertices are likely to be bad keypoint matches.

    For each keypoint pair, the algorithm calculates the mean divergence of
    its incident triangles. The keypoint pair with the highest mean
    divergence is eliminated and the process starts over with a new Delaunay
    triangulation. This continues until the mean divergence for all keypoint
    pairs is below a threshold (to allow for a little bit of noise and
    warping) or until the number of keypoint pairs is reduced below four.
    """
    log: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    assert len(keypoints1) == len(keypoints2)
    n_iterations = 0
    while True:
        if len(keypoints1) < 4:
            break  # Three or fewer keypoints; no point in further elimination.
        total_divergence = np.zeros(len(keypoints1), dtype=float)
        n_incident_triangles = np.zeros(len(keypoints1), dtype=int)
        triangulation = Delaunay(keypoints1)
        log.append((keypoints1, keypoints2, is_spurious, triangulation.simplices))
        for simplex in triangulation.simplices:
            triangle1 = keypoints1[simplex]
            triangle2 = keypoints2[simplex]
            equilateralness1 = calculate_equilateralness(triangle1)
            equilateralness2 = calculate_equilateralness(triangle2)
            divergence = abs(np.log(equilateralness1 / equilateralness2))
            total_divergence[simplex] += divergence
            n_incident_triangles[simplex] += 1
        mean_divergence = total_divergence / n_incident_triangles
        n_vertices_over_threshold = (mean_divergence > elimination_threshold).sum()
        if n_vertices_over_threshold == 0:
            break  # All divergences below threshold; exit loop.
        most_divergent_vertex = np.argmax(mean_divergence)
        keypoints1 = np.delete(keypoints1, most_divergent_vertex, axis=0)
        keypoints2 = np.delete(keypoints2, most_divergent_vertex, axis=0)
        is_spurious = np.delete(is_spurious, most_divergent_vertex, axis=0)
        n_iterations += 1
    print(f"Iterations: {n_iterations}")
    print(f"Remaining points: {len(keypoints1)}")
    print(f"Remaining spurious points: {(is_spurious == True).sum()}")
    return keypoints1, keypoints2, log
