import numpy as np


class City:

    def __init__(self, coords: tuple, city_index) -> None:
        self.coords = coords
        self.city_index = city_index


def create_city_map(country_size=1000, num_cities=5):
    rng = np.random.default_rng(12345)

    # Create a matrix that represents tha map
    # There's no need to create the matrix, just need to generate the coords for the city and set some rules
    # map_matrix = np.zeros([country_size, country_size])

    cities = {}
    # populate the map with cities
    for i in range(num_cities):
        x = rng.integers(low=0, high=country_size)
        y = rng.integers(low=0, high=country_size)
        # print(f"x = {x}, y = {y}")
        # map_matrix[x,y] = City(tuple(x,y), i)
        city_name = f"city{i}"
        cities[city_name] = {"coords": tuple((x, y))}

    # Create a distance matrix for the problem
    city_matrix = np.zeros([num_cities, num_cities])

    print(cities)

    # calculate the distance from each city to the next city. Unit difference in x-pos is 100 miles, in y-pos is 10 miles
    # for i in range(num_cities):
    # index = (city_matrix[:0] == i).argmax()
    # print(f"Index = {index}")
    # print(city_matrix[]

    return city_matrix


# Geneerate a distance matrix for n cities and populate with random distances
def generate_distance_matrix(n):
    rng = np.random.default_rng(12345)
    matrix = np.zeros([n, n])

    for a, b in np.ndindex(matrix.shape):
        if a != b:
            matrix[a, b] = rng.integers(low=100, high=3000)
        else:
            matrix[a, b] = 0

    return matrix


create_city_map(20, 5)
# how to make a representative data set for TSP? and how to visualize it?
