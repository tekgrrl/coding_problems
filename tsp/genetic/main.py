"""This module implements a genetic algorithm to solve the Travelling Salesman Problem."""

import select
import numpy as np
import matplotlib.pyplot as plt
from tsp.utils.dst_matrx_helpers import (
    generate_distance_matrix,
    visualize_distance_matrix,
)


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


def determine_fitness(chromosome, city_matrix):
    """
    Calculate the fitness of a chromosome in the population.
    The fitness is the reciprocal of the total distance of the tour.
    """

    total_distance = 0
    # There are only n-1 edges in a tour
    for i in range(len(chromosome) - 1):
        total_distance += city_matrix[chromosome[i], chromosome[i + 1]]
    fitness = 1 / total_distance

    return fitness


def get_fitnesses(population, city_matrix):
    """
    Calculate the fitness of each chromosome in the population.
    """
    fitnesses = []
    for chromosome in population:
        fitness = determine_fitness(chromosome, city_matrix)
        fitnesses.append(fitness)
    return fitnesses


def create_roulette_wheel(population, fitnesses):
    """
    Create a roulette wheel for selecting chromosomes based on their fitness.
    This both normalizes the fitness values and creates a cumulative sum array.
    """
    total_fitness = sum(fitnesses)
    wheel = np.cumsum([fitness / total_fitness for fitness in fitnesses])
    return wheel


def select_chromosome(roulette_wheel, population):
    """
    Select a chromosome from the population based on the roulette wheel.
    """
    rng = np.random.default_rng()  # unseeded to provide random results
    random_number = rng.random()
    for i, value in enumerate(roulette_wheel):
        if random_number < value:
            return population[i]


def crossover(parent1, parent2):
    """
    Perform crossover between two parents to create two offspring.
    """
    rng = np.random.default_rng()
    crossover_point = rng.integers(1, len(parent1))

    # Create copies of the parents to avoid modifying them during the list comprehension
    parent1_copy = parent1.copy()
    parent2_copy = parent2.copy()

    # Create the children by taking the first part from one parent and the remaining unique elements from the other parent
    child1 = np.append(
        parent1_copy[:crossover_point],
        [city for city in parent2_copy if city not in parent1_copy[:crossover_point]],
    )
    child2 = np.append(
        parent2_copy[:crossover_point],
        [city for city in parent1_copy if city not in parent2_copy[:crossover_point]],
    )

    return child1, child2


def two_point_crossover(parent1, parent2):
    """
    Perform 2 point crossover between two parents to create two offspring.
    """
    size = len(parent1)
    rng = np.random.default_rng()
    cx1, cx2 = sorted(rng.choice(size, 2, replace=False))
    # Initialize children as None lists or with invalid markers
    child1 = [None] * size
    child2 = [None] * size

    # Insert the slices from each parent into the opposite child
    child1[cx1 : cx2 + 1], child2[cx1 : cx2 + 1] = (
        parent2[cx1 : cx2 + 1],
        parent1[cx1 : cx2 + 1],
    )

    # Function to fill remaining spots in children
    def fill_child(child, parent, cx1, cx2):
        fill_pos = (cx2 + 1) % size  # Start immediately after the copied segment
        for gene in parent:
            if gene not in child:  # Only add if not already in the child
                child[fill_pos] = gene
                fill_pos = (fill_pos + 1) % size  # Move to next position, wrap around

    # Fill in the remaining positions in children
    fill_child(child1, parent1, cx1, cx2)
    fill_child(child2, parent2, cx1, cx2)

    return child1, child2


def run_genetic_algorithm(population_size, city_matrix, generations):
    """
    Run a genetic algorithm to solve the Travelling Salesman Problem.
    Args:
        population_size (int): The size of the population.
        city_matrix (numpy.ndarray): The matrix representing the distances between cities.
        generations (int): The number of generations to run the algorithm.
    Returns:
        list: A list of the best fitness values for each generation.
    """
    population = create_population(population_size, city_matrix)
    fitnesses = get_fitnesses(population, city_matrix)
    best_fitnesses = [max(fitnesses)]
    for _ in range(generations):
        roulette_wheel = create_roulette_wheel(population, fitnesses)
        new_population = []
        for _ in range(population_size // 2):
            parent1 = select_chromosome(roulette_wheel, population)
            parent2 = select_chromosome(roulette_wheel, population)
            child1, child2 = two_point_crossover(parent1, parent2)
            new_population.extend([child1, child2])
        population = new_population
        fitnesses = get_fitnesses(population, city_matrix)
        best_fitnesses.append(max(fitnesses))
    return best_fitnesses


if __name__ == "__main__":
    city_matrix_data = generate_distance_matrix(10)
    # population = create_population(5, city_matrix_data)

    # fitnesses = get_fitnesses(population, city_matrix_data)

    # # Roulette wheel selection
    # roulette_wheel = create_roulette_wheel(population, fitnesses)
    # selection1 = select_chromosome(roulette_wheel, population)
    # selection2 = select_chromosome(roulette_wheel, population)
    # print(selection1, selection2)
    # crossover_result = two_point_crossover(selection1, selection2)
    result = run_genetic_algorithm(10, city_matrix_data, 10)
    print(result)
