"""This module implements a genetic algorithm to solve the Travelling Salesman Problem."""

import numpy as np
from tsp.utils.dst_matrx_helpers import generate_distance_matrix


# Notes:
#   Implementation of a genetic algorithm to solve the Travelling Salesman Problem
#   How do I access the data generator in a different project?
#   Chromosomes will be represented as Lists of Integers


def create_population(population_size, city_matrix):
    """cd ..
    Create a population of random chromosomes.

    Args:
        population_size (int): The size of the population.
        city_matrix (numpy.ndarray): The matrix representing the distances between cities.

    Returns:
        list: A list of randomly generated chromosomes.

    """
    rng = np.random.default_rng(12345)
    population = []
    for i in range(population_size):
        chromosome = rng.permutation(len(city_matrix))
        population.append(chromosome)
    return population


city_matrix_data = generate_distance_matrix(10)
chromosomes = create_population(10, city_matrix_data)

print(chromosomes)
