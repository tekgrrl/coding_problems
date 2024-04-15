"""This module implements a greedy algorithm to solve the Travelling Salesman Problem."""

import numpy as np
from tsp.utils.dst_matrx_helpers import generate_distance_matrix


def solve_tsp(N):
    """
    Solve the Traveling Salesman Problem (TSP) using a heuristic approach.

    Args:
      N (int): Number of cities.

    Returns:
      float: Shortest tour distance in miles.

    The TSP is a classic optimization problem in computer science and operations research.
    Given a set of n cities (or nodes) and the distances between each pair of cities,
    the objective is to find the permutation (or order) of visiting the cities that minimizes the total distance traveled,
    subject to the constraints of visiting each city exactly once, returning to the starting city, and minimizing the total distance.

    This function uses a greedy approach to find the optimal tour.
    It starts with a random city and iteratively selects the closest unvisited city until all cities have been visited.
    The starting city can affect the final tour length, so the TSP is often solved multiple times with different starting cities,
    and the best tour is selected from the results.

    The function generates a distance matrix using the `generate_distance_matrix` function from the `tsp_map_gen` module.
    The distance matrix represents the distances between each pair of cities.

    The function returns the shortest tour distance in rounded miles.
    """

    city_distance_matrix = generate_distance_matrix(N)

    all_distances = []

    # Outer loop, do for every city in order
    for i in range(N):
        # Per loop setup
        current_city = i
        visited_cities = []
        total_distance = 0
        # Inner loop, do for every city except the last
        for _ in range(N - 1):
            row = np.copy(
                city_distance_matrix[current_city]
            )  # Deep copy of the row of distances for the currebt city
            row[row == 0] = (
                np.inf
            )  # Set the diagonal to infinity to avoid selecting the same city
            row[visited_cities] = (
                np.inf
            )  # Set the visited cities to infinity to avoid selecting them
            closest_city = np.argmin(row)  # Find the index of the closest city
            min_distance = row[closest_city]  # Find the distance to the closest city

            total_distance = total_distance + min_distance
            visited_cities.append(current_city)
            current_city = closest_city

        all_distances.append(total_distance)

    min_distance = min(all_distances)

    return min_distance


if __name__ == "__main__":
    N = 1000  # Number of cities
    shortest_distance = solve_tsp(N)
    print(f"Shortest tour distance: {round(shortest_distance * 100)} miles")
