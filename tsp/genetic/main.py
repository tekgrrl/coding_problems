"""This module implements a genetic algorithm to solve the Travelling Salesman Problem."""

from json import tool
import numpy as np
import matplotlib.pyplot as plt
import random
from tsp.utils.dst_matrx_helpers import (
    generate_distance_matrix,
    visualize_distance_matrix,
)

from tsp.greedy.main import solve_tsp
from tsp.utils.tsp import TravelingSalesmanProblem


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
    rng = np.random.default_rng()  # should use the same random seed everywhere
    population = []
    for i in range(population_size):
        chromosome = rng.permutation(len(city_matrix))
        population.append(chromosome)

    return population


def create_bad_population(population_size, city_matrix):
    """
    This will create a population deliberately skewed towards bad chromosomes
    It will be a smaller population than for the normal flow so
    adjust the population size accordingly
    """
    # Create a population of chromosomes with the worst fitness values that we can find
    rng = np.random.default_rng(12345)
    temp_population = []
    bad_population = []

    for i in range(population_size):
        chromosome = rng.permutation(len(city_matrix))
        temp_population.append(chromosome)

    overall_fitness = get_fitnesses(temp_population, city_matrix)

    # calculate the sum of the fitness values and then normalize
    total_fitness = sum(overall_fitness)
    normalized_fitness = [fitness / total_fitness for fitness in overall_fitness]

    # find the mean fitness value
    mean_fitness = np.mean(normalized_fitness)

    # find the worst fitness values and add them to bad_population
    for i in range(len(normalized_fitness)):
        if normalized_fitness[i] < mean_fitness:
            bad_population.append(temp_population[i])

    return bad_population


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
    # print(f"Random Number: {random_number}, Selected Index: {index}")

    return population[index]


def stochastic_universal_sampling(roulette_wheel, population, rng):
    random_number = rng.random()
    # Generate four more numbers that are equally spaced between 0 and 1
    numbers = np.linspace(random_number, 1, 5)[1:]


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
    mutation_frequency = 0.2  # TUNABLE PARAMETER
    if rng.random() < mutation_frequency:  # Mutation chance
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

    # Slices and mutatez
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


def find_fittest_chromosome_in_population(population, city_matrix):
    """
    Find the fittest chromosome in the population.
    """
    fitnesses = [
        determine_fitness(chromosome, city_matrix) for chromosome in population
    ]
    fittest_chromosome_index = np.argmax(fitnesses)
    return population[fittest_chromosome_index]


def find_least_fit_chromosome_in_population(population, city_matrix):
    """
    Find the least fit chromosome in the population.
    """
    fitnesses = [
        determine_fitness(chromosome, city_matrix) for chromosome in population
    ]
    least_fit_chromosome_index = np.argmin(fitnesses)
    return population[least_fit_chromosome_index]


# TODO This standalone function is not as good as the separate two_point_crossover()
# and mutate() functions. They are different, not just separate
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
    return diversity_ratio


def promote_unique_elite(elite_chromosome, population, city_matrix):

    similarity_threshold = len(elite_chromosome) * 0.1  # TUNABLE PARAMETER

    # Convert chromosomes to tuples for easy comparison
    population_set = {tuple(chromosome) for chromosome in population}
    elite_tuple = tuple(elite_chromosome)

    for individual in population:
        if permutation_distance(elite_chromosome, individual) < similarity_threshold:
            return False  # Do not promote this elite as it's not unique enough

        # Replace the least fit individual with the unique elite
    least_fit_idx = np.argmin(
        [determine_fitness(chromosome, city_matrix) for chromosome in population]
    )
    population[least_fit_idx] = elite_chromosome
    return True  # Elite was promoted


def replace_with_elites(new_population, old_population, city_matrix, elite_count, rng):
    """
    Replaces a portion of the new population with the elite individuals from the old population.

    Args:
        new_population (list): The new population of individuals.
        old_population (list): The old population of individuals.
        city_matrix (list): The matrix representing the distances between cities.
        elite_count (int): The number of elite individuals to replace.
        rng (random.Random): The random number generator.

    Returns:
        list: The updated population with elite individuals replaced.

    """

    sorted_old_population = sorted(
        old_population, key=lambda x: determine_fitness(x, city_matrix), reverse=True
    )

    elites_promoted = 0
    for i in range(elite_count):
        if promote_unique_elite(sorted_old_population[i], new_population, city_matrix):
            elites_promoted += 1

    print(f"Promoted {elites_promoted} elite individuals to the new population")
    # Ensure the new population is shuffled to maintain genetic diversity
    rng.shuffle(new_population)
    return new_population


def run_genetic_algorithm(city_matrix, logs):
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
    MAX_GENS = 500
    rng = np.random.default_rng(12345)
    population_size = city_matrix.shape[0] * 2  # TUNABLE PARAMETER
    population = create_population(
        population_size, city_matrix
    )  # Create the initial population
    fitnesses = get_fitnesses(population, city_matrix)

    elite_count = 20  # 16 works great for 20 cities

    generation_count = 0  # setup for the loops
    last_five_populations = []  # list to store the last five populations
    for _ in range(MAX_GENS):
        roulette_wheel = create_roulette_wheel(population, fitnesses)
        new_population = []
        for _ in range(population_size // 2):
            parent1 = select_chromosome(roulette_wheel, population, rng)
            parent2 = select_chromosome(roulette_wheel, population, rng)
            while np.array_equal(parent1, parent2):
                parent2 = select_chromosome(roulette_wheel, population, rng)
            child1, child2 = two_point_crossover(parent1, parent2, rng)
            new_population.extend([child1, child2])

        new_population = replace_with_elites(
            new_population,
            population,
            city_matrix,
            elite_count,
            rng=rng,
        )
        population = new_population
        fitnesses = get_fitnesses(population, city_matrix)

        diversity_score = analyze_diversity(population, city_matrix)
        diversity_ratio = check_diversity(population)

        print(
            f"Population {generation_count} Diversity Score: {diversity_score} Diversity Ratio: {diversity_ratio}"
        )

        # Add the current population to the list of last five populations
        last_five_populations.append(population)
        # If we have more than 5 populations in the list, remove the oldest one
        if len(last_five_populations) > 5:
            last_five_populations.pop(0)

        logs["best_fitness"].append(np.max(fitnesses))
        logs["worst_fitness"].append(np.min(fitnesses))
        logs["average_fitness"].append(np.mean(fitnesses))

        generation_count += 1
        # check stopping condition, i.e.: there has been no improvement in the best fitness for the last 10 generations
        if len(logs["best_fitness"]) > 10:
            if len(set(logs["best_fitness"][-10:])) == 1:
                break
    # Add the last five populations to the logs
    logs["last_five_populations"] = last_five_populations

    # Handle stopping condition
    if generation_count == MAX_GENS:
        print("Maximum generations reached")
    else:
        print(f"Converged after {generation_count} generations")
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


def display_results(label, result, logs=None, num_gens=None):
    if num_gens is not None:
        print(f"{label} algorithm results (num_gens={num_gens}):")
    else:
        print(f"{label} algorithm results:")
    print(f"Best chromosome: {result['chromosome']}")
    print(f"Total distance: {result['total_distance']}")
    print()

    if logs is not None:
        plt.figure(figsize=(10, 5))
        plt.plot(logs["best_fitness"], label="Best Fitness")
        plt.plot(logs["average_fitness"], label="Average Fitness")
        plt.plot(logs["worst_fitness"], label="Worst Fitness")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Fitness over Generations")
        plt.legend()
        plt.show()


def permutation_distance(chromosome1, chromosome2):
    # For TSP, a simple measure could be the count of different positions
    return sum(c1 != c2 for c1, c2 in zip(chromosome1, chromosome2))


def analyze_diversity(population, city_matrix):
    # Sort population based on fitness or any other criterion
    sorted_population = sorted(
        population, key=lambda x: determine_fitness(x, city_matrix)
    )
    # Calculate distances between consecutive chromosomes
    distances = [
        permutation_distance(sorted_population[i], sorted_population[i + 1])
        for i in range(len(sorted_population) - 1)
    ]
    # Average distance is a measure of diversity
    average_distance = sum(distances) / len(distances)
    return average_distance


if __name__ == "__main__":

    random_seed = 200  # TUNABLE PARAMETER

    # Randomly generated city distance matrix
    # city_matrix_data = generate_distance_matrix(num_cities, random_seed)

    # Using TSPLIB data
    tsp = TravelingSalesmanProblem("bayg29")  # optimal distance 9074.147
    city_matrix_data = np.array(tsp.distances)

    logs = {"best_fitness": [], "worst_fitness": [], "average_fitness": []}

    # visualize_distance_matrix(city_matrix_data)
    greedy_result = solve_tsp(city_matrix_data)
    display_results("Greedy", greedy_result)

    genetic_result = run_genetic_algorithm(city_matrix_data, logs)
    display_results("Genetic", genetic_result, logs)

    # Assuming you have a current_population variable containing the current generation's population
    # for i in range(len(logs["last_five_populations"])):
    #     current_population = logs["last_five_populations"][i]
    #     diversity_score = analyze_diversity(current_population, city_matrix_data)
    #     print(f"Population {i} Diversity Score: {diversity_score}")
