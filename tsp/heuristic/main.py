

"""
Given:

  - A set of n cities (or nodes)
  - The distances between each pair of cities

The objective is to find the permutation (or order) of visiting the cities that minimizes the total distance traveled, subject to the following constraints:

  1. Visit each city exactly once: 
            The tour must visit each city in the set exactly one time, without revisiting any city before completing the tour.
  2. Return to the starting city: 
            After visiting all the cities, the tour must return to the initial starting city.
  3. Minimum total distance: 
            The sum of the distances between consecutive cities in the tour should be minimized, resulting in the shortest possible overall route.
 """

