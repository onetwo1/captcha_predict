import os
import sys
from operator import attrgetter
import random
import bisect

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
if str(CURRENT_PATH) not in sys.path:
    sys.path.append(CURRENT_PATH)
import gaps_utils as utils
from crossover import Crossover
from image_analysis import ImageAnalysis
from individual import Individual


def roulette_selection(population, elites=4):
    fitness_values = [individual.fitness for individual in population]
    probability_intervals = [
        sum(fitness_values[: i + 1]) for i in range(len(fitness_values))
    ]

    def select_individual():
        """Selects random individual from population based on fitess value"""
        random_select = random.uniform(0, probability_intervals[-1])
        selected_index = bisect.bisect_left(probability_intervals, random_select)
        return population[selected_index]

    selected = []
    for i in range(len(population) - elites):
        first, second = select_individual(), select_individual()
        selected.append((first, second))

    return selected

class GeneticAlgorithm(object):
    TERMINATION_THRESHOLD = 10

    def __init__(self, image, piece_size, population_size, generations, elite_size=2):
        self._image = image
        self._piece_size = piece_size
        self._generations = generations
        self._elite_size = elite_size
        pieces, rows, columns = utils.flatten_image(image, piece_size, indexed=True)
        self._population = [
            Individual(pieces, rows, columns) for _ in range(population_size)
        ]
        self._pieces = pieces

    def start_evolution(self, verbose):


        ImageAnalysis.analyze_image(self._pieces)

        fittest = None
        best_fitness_score = float("-inf")
        termination_counter = 0

        for generation in range(self._generations):
            new_population = []

            # Elitism
            elite = self._get_elite_individuals(elites=self._elite_size)
            new_population.extend(elite)

            selected_parents = roulette_selection(
                self._population, elites=self._elite_size
            )

            for first_parent, second_parent in selected_parents:
                crossover = Crossover(first_parent, second_parent)
                crossover.run()
                child = crossover.child()
                new_population.append(child)

            fittest = self._best_individual()

            if fittest.fitness <= best_fitness_score:
                termination_counter += 1
            else:
                best_fitness_score = fittest.fitness

            if termination_counter == self.TERMINATION_THRESHOLD:
                return fittest

            self._population = new_population

        return fittest

    def _get_elite_individuals(self, elites):
        """Returns first 'elite_count' fittest individuals from population"""
        return sorted(self._population, key=attrgetter("fitness"))[-elites:]

    def _best_individual(self):
        """Returns the fittest individual from population"""
        return max(self._population, key=attrgetter("fitness"))
