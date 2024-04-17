"""Module for generating a distance matrix for the Traveling Salesman Problem"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class TSPMapGen:
    """
    Class for generating a distance matrix for the Traveling Salesman Problem.

    Attributes:
        space_size (int): Space size (n x n).
        strata_grid_size (int): Strata grid size (m x m).
        decay_factor (float): Decay factor for density values.
        density_increase (int): Factor by which a stratum's density increases when a point is added.
        neighbor_weight (float): Weight of neighbors' density in the selectability quotient.
        strata_density (numpy.ndarray): Density matrix.
        rng (numpy.random.Generator): Random number generator.
    """

    def __init__(
        self,
        space_size,
        rng_seed,
    ):
        self.space_size = space_size
        self.strata_grid_size = 5
        self.decay_factor = 0.95
        self.density_increase = 2
        self.neighbor_weight = 0.5
        self.strata_density = np.ones((self.strata_grid_size, self.strata_grid_size))
        self.rng = np.random.default_rng(rng_seed)

    def calculate_average_neighbor_density(self, i, j) -> np.float64:
        """
        Calculates the average density of neighbors for a given stratum.

        Args:
            i (int): The row index of the stratum.
            j (int): The column index of the stratum.

        Returns:
            np.float64: The average density of neighbors.
        """
        neighbors = []
        directions = [-1, 0, 1]

        for di in directions:
            for dj in directions:
                if (
                    0 <= i + di < self.strata_grid_size
                    and 0 <= j + dj < self.strata_grid_size
                ):
                    neighbors.append(self.strata_density[i + di, j + dj])
        return np.mean(neighbors)

    def calculate_selectability(self):
        """
        Calculates the selectability quotient for each stratum based on its own density,
        the average density of its neighbors, and applies a decay rate.

        Returns:
            numpy.ndarray: A 2D array representing the selectability quotient for each stratum.
        """
        strata_size = self.strata_grid_size

        selectability = np.zeros((strata_size, strata_size))
        for i in range(strata_size):
            for j in range(strata_size):
                average_neighbor_density = self.calculate_average_neighbor_density(i, j)
                selectability[i, j] = 1 / (
                    self.strata_density[i, j]
                    + self.neighbor_weight * average_neighbor_density
                )
        return selectability

    def place_point(self):
        """
        Places a point in the space by selecting a stratum based on the selectability quotient,
        then updates the density of the selected stratum and applies decay to all.
        Returns the coordinates of the placed point.
        """
        selectability = self.calculate_selectability()
        probabilities = selectability.flatten() / np.sum(selectability)
        chosen_stratum = self.rng.choice(
            np.arange(self.strata_grid_size * self.strata_grid_size), p=probabilities
        )
        chosen_i, chosen_j = divmod(chosen_stratum, self.strata_grid_size)
        point_x = chosen_i * (
            self.space_size / self.strata_grid_size
        ) + self.rng.uniform(0, self.space_size / self.strata_grid_size)
        point_y = chosen_j * (
            self.space_size / self.strata_grid_size
        ) + self.rng.uniform(0, self.space_size / self.strata_grid_size)
        self.strata_density[chosen_i, chosen_j] *= self.density_increase
        self.strata_density = np.maximum(1, self.strata_density * self.decay_factor)
        return (point_x, point_y)

    def calculate_distance_matrix(self, points):
        """
        Calculates a distance matrix for a list of points.

        Args:
            points (list): A list of points.

        Returns:
            numpy.ndarray: A 2D array representing the distance matrix.
        """
        x_coords = np.array([p[0] for p in points])
        y_coords = np.array([p[1] for p in points])
        x_diff = x_coords[:, np.newaxis] - x_coords
        y_diff = y_coords[:, np.newaxis] - y_coords
        distance_matrix = np.sqrt(x_diff**2 + y_diff**2)
        distance_matrix = np.round(distance_matrix * 100, 2)
        return distance_matrix


def visualize_distance_matrix(distance_matrix):
    """
    Visualizes a distance matrix using a heatmap.

    Args:
        distance_matrix (numpy.ndarray): A 2D numpy array containing the distances between points.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        distance_matrix,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        square=True,
        linewidths=0.5,
    )
    plt.title("Distance Matrix Heatmap")
    plt.xlabel("Point Index")
    plt.ylabel("Point Index")
    plt.show()


def generate_distance_matrix(size=10, rng_seed=42):
    """
    Generates a distance matrix for a given number of points.

    Args:
        size (int): The number of points to generate.

    Returns:
        numpy.ndarray: A 2D array representing the distance matrix.
    """
    tps_map_gen = TSPMapGen(size, rng_seed)
    points = []
    for _ in range(size):
        point = tps_map_gen.place_point()
        points.append(point)

    matrix = tps_map_gen.calculate_distance_matrix(points)
    print("Matrix generated successfully!")
    return matrix


def test_triangle_inequality(distance_matrix):
    """
    Tests the triangle inequality for three random points in the distance matrix.

    Args:
        distance_matrix (numpy.ndarray): A 2D numpy array containing the distances between points.

    Returns:
        bool: True if the triangle inequality holds for the three points, False otherwise.
    """
    size = distance_matrix.shape[0]
    rng = np.random.default_rng()
    points = rng.choice(size, 3, replace=False)
    a, b, c = points
    return (
        distance_matrix[a, b] + distance_matrix[b, c] >= distance_matrix[a, c]
        and distance_matrix[a, c] + distance_matrix[c, b] >= distance_matrix[a, b]
        and distance_matrix[b, a] + distance_matrix[a, c] >= distance_matrix[b, c]
    )


def analyze_distance_matrix(distance_matrix):
    """
    Analyzes a distance matrix by performing statistical analysis and plotting the distribution of distances.

    Parameters:
    - distance_matrix (numpy.ndarray): The distance matrix to be analyzed.

    Returns:
    - stats (dict): A dictionary containing the statistical analysis results, including mean, standard deviation,
                    minimum, and maximum distances.

    """
    # Flatten the matrix and filter out zeros and redundant values (upper triangle)
    triu_indices = np.triu_indices_from(distance_matrix, k=1)
    distances = distance_matrix[triu_indices]

    # Statistical analysis
    stats = {
        "mean": np.mean(distances),
        "std": np.std(distances),
        "min": np.min(distances),
        "max": np.max(distances),
    }

    # Plotting the distribution of distances
    plt.figure(figsize=(10, 5))
    sns.histplot(distances, kde=True)
    plt.title("Distribution of Distances")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.show()

    return stats


def calculate_uniformity(distance_matrix):
    """
    Calculates the uniformity score of a distance matrix.
    Crude implementation of the coefficient of variation (CV) for the distances.

    Not very useful currently
    """
    triu_indices = np.triu_indices_from(distance_matrix, k=1)
    distances = distance_matrix[triu_indices]
    mean_distance = np.mean(distances)
    std_deviation = np.std(distances)
    cv = std_deviation / mean_distance if mean_distance != 0 else 0
    uniformity_score = (
        1 - cv
    )  # Inverse of CV for a 'uniformity' metric (higher is more uniform)
    return uniformity_score


if __name__ == "__main__":
    dmatrix = generate_distance_matrix(10, 100)
    # print(dmatrix)

    print(f"triangle equality? = {test_triangle_inequality(dmatrix)}")
    # visualize_distance_matrix(dmatrix)
    stats = analyze_distance_matrix(dmatrix)
    print(stats)
    uniformity_score = calculate_uniformity(dmatrix)
    print(f"Matrix uniformity score = {uniformity_score:.2f}")
