"""This module implements a greedy algorithm to solve the Travelling Salesman Problem."""

import numpy as np
from tsp.utils.dst_matrx_helpers import (
    generate_distance_matrix,
    visualize_distance_matrix,
)


def solve_tsp(city_distance_matrix):
    """
    Solve the Traveling Salesman Problem (TSP) using a heuristic approach.

    Args:
        city_distance_matrix (numpy.ndarray): A 2D array representing the distances between each pair of cities.
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

    results = {}
    num_cities = city_distance_matrix.shape[0]
    # Outer loop, do for every city in order
    for i in range(num_cities):
        # Per loop setup
        current_city = i
        visited_cities = []
        distances = []
        total_distance = 0
        # Inner loop, do for every city except the last
        for city_count in range(num_cities):
            # halting condition. If we are at the last city, we need to return to the starting city
            if city_count == num_cities - 1:
                visited_cities.append(current_city)
                total_distance = (
                    total_distance + city_distance_matrix[current_city, i]
                )  # Return to the starting city given by column i of the row of the current city
                break
            row = np.copy(
                city_distance_matrix[current_city]
            )  # Deep copy of the row of distances for the current city
            row[row == 0] = (
                np.inf
            )  # Set the diagonal to infinity to avoid selecting the same city
            row[visited_cities] = (
                np.inf
            )  # Set the visited cities to infinity to avoid selecting them
            closest_city = np.argmin(row)  # Find the index of the closest city
            closest_city_distance = row[
                closest_city
            ]  # Find the distance to the closest city

            distances.append(closest_city_distance)  # for logging

            total_distance = total_distance + closest_city_distance
            visited_cities.append(
                current_city
            )  # Add the current city to the list of visited cities
            current_city = closest_city

        distances.append(city_distance_matrix[current_city, i])  # for logging

        results[i] = {
            "chromosome": visited_cities,
            "total_distance": total_distance,
        }

    min_distance_index = min(results, key=lambda x: results[x]["total_distance"])

    return results[min_distance_index]


if __name__ == "__main__":
    N = 10  # Number of cities
    cities = generate_distance_matrix(N)
    candidate_chromosome = solve_tsp(N, cities)
    print(
        f"""The chromosome with the shortest route is 
        {candidate_chromosome['visited_cities']} 
        with {candidate_chromosome['total_distance']} miles"""
    )
    visualize_distance_matrix(cities)
