from collections import namedtuple
from itertools import combinations
import numpy as np
from scipy.spatial import Delaunay

KeypointReductionResult = namedtuple(
    "KeypointReductionResult",
    ["keypoints1", "keypoints2", "is_spurious", "initial_simplices", "final_simplices"],
)


def get_normalized_side_lengths(triangle: np.ndarray) -> np.ndarray:
    assert triangle.shape == (3, 2)
    side_lengths = np.array(
        [np.linalg.norm(vertex1 - vertex2) for vertex1, vertex2 in combinations(triangle, 2)]
    )
    return side_lengths / side_lengths.sum()


def calculate_shape_discrepancy(triangle1: np.ndarray, triangle2: np.ndarray) -> float:
    normalized_side_lengths1 = get_normalized_side_lengths(triangle1)
    normalized_side_lengths2 = get_normalized_side_lengths(triangle2)
    return np.sqrt(((normalized_side_lengths1 - normalized_side_lengths2) ** 2).sum())


def reduce_to_delaunay_consistency(
    keypoints1: np.ndarray,
    keypoints2: np.ndarray,
    is_spurious: np.ndarray,
    elimination_threshold: float = 0.05,
) -> KeypointReductionResult:
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
    assert len(keypoints1) == len(keypoints2)
    n_iterations = 0
    initial_simplices: np.ndarray | None = None
    while True:
        if len(keypoints1) < 4:
            if initial_simplices is not None:
                triangulation = Delaunay(keypoints1)
            break  # Three or fewer keypoints; no point in further elimination.
        total_vertex_discrepancy = np.zeros(len(keypoints1), dtype=float)
        n_incident_triangles = np.zeros(len(keypoints1), dtype=int)
        triangulation = Delaunay(keypoints1)
        if initial_simplices is None:
            initial_simplices = triangulation.simplices
        for simplex in triangulation.simplices:
            triangle1 = keypoints1[simplex]
            triangle2 = keypoints2[simplex]
            triangle_discrepancy = calculate_shape_discrepancy(triangle1, triangle2)
            total_vertex_discrepancy[simplex] += triangle_discrepancy
            n_incident_triangles[simplex] += 1
        mean_vertex_discrepancy = total_vertex_discrepancy / n_incident_triangles
        n_vertices_over_threshold = (mean_vertex_discrepancy > elimination_threshold).sum()
        if n_vertices_over_threshold == 0:
            break  # All divergences below threshold; exit loop.
        most_divergent_vertex = np.argmax(mean_vertex_discrepancy)
        keypoints1 = np.delete(keypoints1, most_divergent_vertex, axis=0)
        keypoints2 = np.delete(keypoints2, most_divergent_vertex, axis=0)
        is_spurious = np.delete(is_spurious, most_divergent_vertex, axis=0)
        n_iterations += 1
    if initial_simplices is None:
        final_simplices = None
    else:
        final_simplices = triangulation.simplices
    print(f"Iterations: {n_iterations}")
    print(f"Remaining points: {len(keypoints1)}")
    print(f"Remaining spurious points: {(is_spurious == True).sum()}")
    return KeypointReductionResult(
        keypoints1, keypoints2, is_spurious, initial_simplices, final_simplices
    )
