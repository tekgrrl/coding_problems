"""This module implements a genetic algorithm to solve the Travelling Salesman Problem."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tsp.greedy.main import solve_tsp
from tsp.utils.tsp import TravelingSalesmanProblem


class TSPGeneticAlgorithm:
    def __init__(self):
        self.rng = np.random.default_rng(12345)

    def create_population(self, population_size, city_matrix):
        """
        Create a population of random chromosomes.
        """
        population = []
        for i in range(population_size):
            chromosome = self.rng.permutation(len(city_matrix))
            population.append(chromosome)

        return population

    def determine_fitness(self, chromosome, city_matrix):
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

    def get_fitnesses(self, population, city_matrix):
        """
        Calculate the fitness of each chromosome in the population.
        """
        fitnesses = []
        for chromosome in population:
            fitness = self.determine_fitness(chromosome, city_matrix)
            fitnesses.append(fitness)
        return fitnesses

    # implement tournament selection
    def tournament_selection(self, population, fitnesses, tournament_size):
        """
        Perform tournament selection to select a chromosome from the population.
        """
        # selected_chromosomes = []
        # for _ in range(len(population)):
        tournament_indices = self.rng.choice(
            len(population), tournament_size, replace=False
        )
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        selected_chromosome = population[winner_index]
        return selected_chromosome

    def create_roulette_wheel(self, population, fitnesses):
        """
        Create a roulette wheel for selecting chromosomes based on their fitness.
        This both normalizes the fitness values and creates a cumulative sum array.
        """
        total_fitness = sum(fitnesses)
        wheel = np.cumsum([fitness / total_fitness for fitness in fitnesses])
        return wheel

    def select_chromosome(self, roulette_wheel, population):
        """
        Select a chromosome from the population based on the roulette wheel.
        """
        random_number = self.rng.random()

        index = np.searchsorted(roulette_wheel, random_number)
        # print(f"Random Number: {random_number}, Selected Index: {index}")

        return population[index]

    def stochastic_universal_sampling(self, roulette_wheel, population):
        """
        Selects chromosomes from the population using the Stochastic Universal Sampling (SUS) method.
        """
        population_size = len(population)
        random_number = self.rng.random()
        # generate a random number between 0 and (1/len(population))
        step_size = 1 / population_size
        random_number *= step_size

        # Generate population_size more numbers that are equally spaced between 0 and 1
        points = [random_number + i * step_size for i in range(population_size)]
        # return the set of chromosomes that are closest to the points
        selected_chromosomes = [
            population[np.searchsorted(roulette_wheel, point)] for point in points
        ]
        return selected_chromosomes

    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parents to create two offspring.
        """
        crossover_point = self.rng.integers(1, len(parent1))

        # Create copies of the parents to avoid modifying them during the list comprehension
        parent1_copy = parent1.copy()
        parent2_copy = parent2.copy()

        # Create the children by taking the first part from one parent and the remaining unique elements from the other parent
        child1 = np.append(
            parent1_copy[:crossover_point],
            [
                city
                for city in parent2_copy
                if city not in parent1_copy[:crossover_point]
            ],
        )
        child2 = np.append(
            parent2_copy[:crossover_point],
            [
                city
                for city in parent1_copy
                if city not in parent2_copy[:crossover_point]
            ],
        )

        return child1, child2

    def mutate(self, slice):
        """
        perform mutations on a slice of a population"""
        mutation_frequency = 0.31  # TUNABLE PARAMETER
        if self.rng.random() < mutation_frequency:  # Mutation chance
            mutation_type = self.rng.choice(2)
            if mutation_type == 0:
                slice = slice[::-1]
            elif len(slice) > 1:
                swap_idx = self.rng.integers(0, len(slice) - 1)
                slice[swap_idx], slice[swap_idx + 1] = (
                    slice[swap_idx + 1],
                    slice[swap_idx],
                )
        return slice

    def two_point_crossover(self, parent1, parent2):
        """
        perform 2 point crossover between two parents to create two offspring.
        """
        size = len(parent1)
        cx1, cx2 = sorted(self.rng.choice(size, 2, replace=False))

        # Slices and mutatez
        slice1 = self.mutate(parent2[cx1 : cx2 + 1])
        slice2 = self.mutate(parent1[cx1 : cx2 + 1])

        child1, child2 = [None] * size, [None] * size
        child1[cx1 : cx2 + 1], child2[cx1 : cx2 + 1] = slice1, slice2

        # Function to fill remaining spots in children
        def fill_child(child, parent, cx1, cx2):
            fill_pos = (cx2 + 1) % size  # Start immediately after the copied segment
            for gene in parent:
                if gene not in child:  # Only add if not already in the child
                    child[fill_pos] = gene
                    fill_pos = (
                        fill_pos + 1
                    ) % size  # Move to next position, wrap around

        fill_child(child1, parent1, cx1, cx2)
        fill_child(child2, parent2, cx1, cx2)

        return child1, child2

    def find_fittest_chromosome_in_population(self, population, city_matrix):
        """
        Find the fittest chromosome in the population.
        """
        fitnesses = [
            self.determine_fitness(chromosome, city_matrix) for chromosome in population
        ]
        fittest_chromosome_index = np.argmax(fitnesses)
        return population[fittest_chromosome_index]

    def find_least_fit_chromosome_in_population(self, population, city_matrix):
        """
        Find the least fit chromosome in the population.
        """
        fitnesses = [
            self.determine_fitness(chromosome, city_matrix) for chromosome in population
        ]
        least_fit_chromosome_index = np.argmin(fitnesses)
        return population[least_fit_chromosome_index]

    def two_point_crossover_with_mutation(self, parent1, parent2):
        """
        Perform 2 point crossover between two parents and do mutation to create two offspring.
        """
        size = len(parent1)
        cx1, cx2 = sorted(self.rng.choice(size, 2, replace=False))
        # Initialize children as None lists or with invalid markers
        child1 = [None] * size
        child2 = [None] * size

        # Insert the slices from each parent into the opposite child
        # Insert the slices from each parent into the opposite child
        slice1, slice2 = parent2[cx1 : cx2 + 1], parent1[cx1 : cx2 + 1]

        # Mutation
        if self.rng.random() < 0.25:  # 25% chance of mutation
            mutation_type = self.rng.choice(2)  # Choose between two types of mutation
            if mutation_type == 0:  # Reverse the slice
                slice1, slice2 = slice1[::-1], slice2[::-1]
            else:  # Swap two adjacent elements
                if len(slice1) > 1:  # Ensure there are at least two elements to swap
                    swap_idx = self.rng.integers(0, len(slice1) - 1)
                    slice1[swap_idx], slice1[swap_idx + 1] = (
                        slice1[swap_idx + 1],
                        slice1[swap_idx],
                    )
                if len(slice2) > 1:  # Ensure there are at least two elements to swap
                    swap_idx = self.rng.integers(0, len(slice2) - 1)
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
                    fill_pos = (
                        fill_pos + 1
                    ) % size  # Move to next position, wrap around

        # Fill in the remaining positions in children
        fill_child(child1, parent1, cx1, cx2)
        fill_child(child2, parent2, cx1, cx2)

        return child1, child2

    def check_diversity(self, population):
        """
        Check the diversity of the population.
        """
        unique_chromosomes = {tuple(chromosome) for chromosome in population}
        diversity_ratio = len(unique_chromosomes) / len(population)
        return diversity_ratio

    def promote_unique_elite(self, elite_chromosome, population, city_matrix):
        """
        Promote a unique elite chromosome to the new population.
        """
        similarity_threshold = len(elite_chromosome) * 0.1  # TUNABLE PARAMETER

        for individual in population:
            if (
                self.permutation_distance(elite_chromosome, individual)
                < similarity_threshold
            ):
                return False  # Do not promote this elite as it's not unique enough

        # Replace the least fit individual with the unique elite
        least_fit_idx = np.argmin(
            [
                self.determine_fitness(chromosome, city_matrix)
                for chromosome in population
            ]
        )
        population[least_fit_idx] = elite_chromosome
        return True  # Elite was promoted

    def replace_with_elites(
        self, new_population, old_population, city_matrix, elite_count
    ):
        """
        Replaces a portion of the new population with the elite individuals from the old population.
        """

        sorted_old_population = sorted(
            old_population,
            key=lambda x: self.determine_fitness(x, city_matrix),
            reverse=True,
        )

        for i in range(elite_count):
            self.promote_unique_elite(
                sorted_old_population[i], new_population, city_matrix
            )

        # Ensure the new population is shuffled to maintain genetic diversity
        self.rng.shuffle(new_population)
        return new_population

    def permutation_distance(self, chromosome1, chromosome2):
        # For TSP, a simple measure could be the count of different positions
        return sum(c1 != c2 for c1, c2 in zip(chromosome1, chromosome2))

    def analyze_diversity(self, population, city_matrix):
        # Sort population based on fitness or any other criterion
        sorted_population = sorted(
            population, key=lambda x: self.determine_fitness(x, city_matrix)
        )
        # Calculate distances between consecutive chromosomes
        distances = [
            self.permutation_distance(sorted_population[i], sorted_population[i + 1])
            for i in range(len(sorted_population) - 1)
        ]
        # Average distance is a measure of diversity
        average_distance = sum(distances) / len(distances)
        return average_distance


def run_genetic_algorithm(city_matrix, logs):
    """
    Run a genetic algorithm to solve the Travelling Salesman Problem.
    Args:
        city_matrix (numpy.ndarray): The matrix representing the distances between cities.
        logs (dict): various log data
    Returns:
        result object. This contains the best chromome and the total distance for that
        chromosome.
    """
    MAX_GENS = 500  # Need to stop at some point

    tspga = TSPGeneticAlgorithm()

    population_size = city_matrix.shape[0] * 2  # TUNABLE PARAMETER
    population = tspga.create_population(
        population_size, city_matrix
    )  # Create the initial population
    fitnesses = tspga.get_fitnesses(population, city_matrix)
    elite_count = 20  # Tune for different datasets

    generation_count = 0  # setup for the loops
    gen_log = []  # list to store the populations for analysis
    for _ in range(MAX_GENS):
        roulette_wheel = tspga.create_roulette_wheel(population, fitnesses)

        new_population = []

        # Stochastic Universal Sampling
        # TODO: only works with even multiples of population size
        # selected_chromosomes = stochastic_universal_sampling(
        #     roulette_wheel, population, rng
        # )
        # for i in range(0, len(selected_chromosomes), 2):
        #     parent1 = selected_chromosomes[i]
        #     parent2 = selected_chromosomes[i + 1]
        #     child1, child2 = two_point_crossover(parent1, parent2, rng)
        #     new_population.extend([child1, child2])

        # Roulette Wheel Selection
        # roulette_wheel = create_roulette_wheel(population, fitnesses)
        # for _ in range(population_size // 2):
        #     parent1 = select_chromosome(roulette_wheel, population)
        #     parent2 = select_chromosome(roulette_wheel, population)
        #     while np.array_equal(parent1, parent2):
        #         parent2 = select_chromosome(roulette_wheel, population)
        #     child1, child2 = two_point_crossover(parent1, parent2)
        #     new_population.extend([child1, child2])

        # use tournament_selection() to fill new_population
        for i in range(population_size):

            tournament_size = 2  # TUNABLE PARAMETER

            parent1 = tspga.tournament_selection(population, fitnesses, tournament_size)
            parent2 = tspga.tournament_selection(population, fitnesses, tournament_size)
            while np.array_equal(parent1, parent2):
                # This was a bug where I copied and pasted the code from the RWS section
                # But it gave me the closest to an optimal solution so it's staying
                parent2 = tspga.select_chromosome(roulette_wheel, population)
            child1, child2 = tspga.two_point_crossover(parent1, parent2)
            new_population.extend([child1, child2])

        new_population = tspga.replace_with_elites(
            new_population, population, city_matrix, elite_count
        )

        population = new_population
        fitnesses = tspga.get_fitnesses(population, city_matrix)

        # Add the current population to the generation log
        gen_log.append(population)

        logs["best_fitness"].append(np.max(fitnesses))
        logs["worst_fitness"].append(np.min(fitnesses))
        logs["average_fitness"].append(np.mean(fitnesses))

        generation_count += 1
        # check stopping condition, i.e.: there has been no improvement in the best fitness for the last 20 generations
        if len(logs["best_fitness"]) > 30:
            if len(set(logs["best_fitness"][-30:])) == 1:
                break
    # Add the last five populations to the logs
    logs["generations_log"] = gen_log

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


def calculate_gene_frequencies_over_generations(generations_log):
    num_genes = len(
        generations_log[0][0]
    )  # Assuming all chromosomes are of the same length.
    num_generations = len(generations_log)
    num_cities = num_genes  # In TSP, the number of cities equals the number of positions in the chromosome.

    # Initialize the frequency array.
    # Rows: number of generations, Columns: number of gene positions.
    gene_frequencies = np.zeros((num_generations, num_genes), dtype=float)

    for generation_index, population in enumerate(generations_log):
        # For each gene position, count the occurrences of each city.
        for position in range(num_genes):
            city_counts = np.zeros(num_cities, dtype=int)
            for chromosome in population:
                city = chromosome[position]
                city_counts[city] += 1
            # The most common city at this position during this generation.
            gene_frequencies[generation_index, position] = np.max(city_counts) / len(
                population
            )

    return gene_frequencies


def plot_gene_frequencies(gene_frequencies):
    # Plot the heatmap
    plt.figure(figsize=(20, 10))  # Adjust figure size as needed
    ax = sns.heatmap(gene_frequencies, cmap="viridis", linewidths=0.1)
    ax.set_title("Gene Frequencies Over Generations")
    ax.set_xlabel("Gene Position (City)")
    ax.set_ylabel("Generation")
    plt.show()


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

    # Assuming you have a generations_log that is a list of populations for each generation
    gene_frequencies = calculate_gene_frequencies_over_generations(
        logs["generations_log"]
    )
    # plot_gene_frequencies(gene_frequencies)
