"""This module implements a genetic algorithm to solve the Travelling Salesman Problem."""

import numpy as np
from tsp.utils.dst_matrx_helpers import (
    generate_distance_matrix,
    visualize_distance_matrix,
)
from tsp.greedy.main import solve_tsp


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
    # These are the 9 edges we can loop over
    for i in range(len(chromosome) - 1):
        total_distance += city_matrix[chromosome[i], chromosome[i + 1]]

    total_distance += city_matrix[
        chromosome[-1], chromosome[0]
    ]  # Add the final edge back to the starting city

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


def select_chromosome(roulette_wheel, population, rng):
    """
    Select a chromosome from the population based on the roulette wheel.
    """
    random_number = rng.random()
    index = np.searchsorted(roulette_wheel, random_number)
    return population[index]


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


def mutate(slice, rng):
    if rng.random() < 0.25:  # Mutation chance
        mutation_type = rng.choice(2)
        if mutation_type == 0:
            slice = slice[::-1]
        elif len(slice) > 1:
            swap_idx = rng.integers(0, len(slice) - 1)
            slice[swap_idx], slice[swap_idx + 1] = slice[swap_idx + 1], slice[swap_idx]
    return slice


def two_point_crossover(parent1, parent2, rng):
    size = len(parent1)
    cx1, cx2 = sorted(rng.choice(size, 2, replace=False))

    # Slices and mutate
    slice1 = mutate(parent2[cx1 : cx2 + 1], rng)
    slice2 = mutate(parent1[cx1 : cx2 + 1], rng)

    child1, child2 = [None] * size, [None] * size
    child1[cx1 : cx2 + 1], child2[cx1 : cx2 + 1] = slice1, slice2

    # Function to fill remaining spots in children
    # TODO break out to separate function
    def fill_child(child, parent, cx1, cx2):
        fill_pos = (cx2 + 1) % size  # Start immediately after the copied segment
        for gene in parent:
            if gene not in child:  # Only add if not already in the child
                child[fill_pos] = gene
                fill_pos = (fill_pos + 1) % size  # Move to next position, wrap around

    fill_child(child1, parent1, cx1, cx2)
    fill_child(child2, parent2, cx1, cx2)

    return child1, child2


def two_point_crossover_with_mutation(parent1, parent2, rng):
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
    # Insert the slices from each parent into the opposite child
    slice1, slice2 = parent2[cx1 : cx2 + 1], parent1[cx1 : cx2 + 1]

    # Mutation
    # TODO break out to separate function
    if rng.random() < 0.25:  # 25% chance of mutation
        mutation_type = rng.choice(2)  # Choose between two types of mutation
        if mutation_type == 0:  # Reverse the slice
            slice1, slice2 = slice1[::-1], slice2[::-1]
        else:  # Swap two adjacent elements
            if len(slice1) > 1:  # Ensure there are at least two elements to swap
                swap_idx = rng.integers(0, len(slice1) - 1)
                slice1[swap_idx], slice1[swap_idx + 1] = (
                    slice1[swap_idx + 1],
                    slice1[swap_idx],
                )
            if len(slice2) > 1:  # Ensure there are at least two elements to swap
                swap_idx = rng.integers(0, len(slice2) - 1)
                slice2[swap_idx], slice2[swap_idx + 1] = (
                    slice2[swap_idx + 1],
                    slice2[swap_idx],
                )

    child1[cx1 : cx2 + 1], child2[cx1 : cx2 + 1] = slice1, slice2

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


def check_diversity(population):
    unique_chromosomes = {tuple(chromosome) for chromosome in population}
    diversity_ratio = len(unique_chromosomes) / len(population)
    print(f"Population Diversity Ratio: {diversity_ratio:.2f}")


def run_genetic_algorithm(population_size, city_matrix, generations):
    """
    Run a genetic algorithm to solve the Travelling Salesman Problem.
    Args:
        population_size (int): The size of the population.
        city_matrix (numpy.ndarray): The matrix representing the distances between cities.
        generations (int): The number of generations to run the algorithm.
    Returns:
        result object. This contains the best chromome and the total distance for that
        chromosome.
    """

    rng = np.random.default_rng(12345)
    population = create_population(
        population_size, city_matrix
    )  # Create the initial population
    fitnesses = get_fitnesses(population, city_matrix)
    for _ in range(generations):
        roulette_wheel = create_roulette_wheel(population, fitnesses)
        new_population = []
        for _ in range(population_size // 2):
            parent1 = select_chromosome(roulette_wheel, population, rng)
            parent2 = select_chromosome(roulette_wheel, population, rng)
            child1, child2 = two_point_crossover(parent1, parent2, rng)
            new_population.extend([child1, child2])
        population = new_population
        check_diversity(population)
        fitnesses = get_fitnesses(population, city_matrix)

    # should return the best chromosome and the total distance for that chromosome
    best_chromosome_idx = np.argmax(fitnesses)
    best_chromosome = population[best_chromosome_idx]
    total_distance = 0
    for i in range(len(best_chromosome) - 1):
        total_distance += city_matrix[best_chromosome[i], best_chromosome[i + 1]]
    total_distance += city_matrix[best_chromosome[-1], best_chromosome[0]]
    return {
        "chromosome": best_chromosome,
        "total_distance": total_distance,
    }


def display_results(label, result, num_gens=None):
    print()
    if num_gens is not None:
        print(f"{label} algorithm results (num_gens={num_gens}):")
    else:
        print(f"{label} algorithm results:")
    print(f"Best chromosome: {result['chromosome']}")
    print(f"Total distance: {result['total_distance']}")


if __name__ == "__main__":
    N = 7  # Number of cities
    num_gens_max = 30

    city_matrix_data = generate_distance_matrix(N, 300)
    # population = create_population(5, city_matrix_data)

    # fitnesses = get_fitnesses(population, city_matrix_data)

    # # Roulette wheel selection
    # roulette_wheel = create_roulette_wheel(population, fitnesses)
    # selection1 = select_chromosome(roulette_wheel, population)
    # selection2 = select_chromosome(roulette_wheel, population)
    # print(selection1, selection2)
    # crossover_result = two_point_crossover(selection1, selection2)
    greedy_result = solve_tsp(N, city_matrix_data)
    display_results("Greedy", greedy_result)

    for num_gens in range(num_gens_max):
        genetic_result = run_genetic_algorithm(N, city_matrix_data, num_gens)
        display_results("Genetic", genetic_result, num_gens)
    # visualize_distance_matrix(city_matrix_data)
