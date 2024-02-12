"""
Solutions to Homework 1 Question 5

Author: Clark Kaminsky
"""

import copy
import numpy as np
from numpy import random
import math


def conflict(row1, col1, row2, col2):
    """
    Would putting two queens in (row1, col1) and (row2, col2) conflict?

	Parameters:
		row1 - int
		col1 - int
		row2 - int
		col2 - int

	Returns:
		True if conflict else False
    """
    return (row1 == row2 or  # same row
            col1 == col2 or  # same column
            row1 - col1 == row2 - col2 or  # same \ diagonal
            row1 + col1 == row2 + col2)  # same / diagonal

def h(state):
    """
	Parameter:
		state - a list of length 'n' for n-queens problem, where the c-th element
		       of the list holds the value for the row number of the queen in
		       column c.
	Returns:
		num_conflicts - int corresponding to number of conflicting queens for a given state
    """
    num_conflicts = 0
    N = len(state)
    for c1 in range(N):
        for c2 in range(c1+1, N):
            num_conflicts += conflict(state[c1], c1, state[c2], c2)
    return num_conflicts

def hill_climbing(init_state):
    """
    Run the steepest-descent hill-climbing algorithm
    :param init_state: a state of length 7 representing the row that the queen on each column occupies
    :return: returns the final state, its cost, the number of steps to reach that state, and the initial state's cost
    """
    init_h = h(init_state)
    current = init_state
    current_val = init_h
    num_steps = 0
    while(True):
        best_neighbor_state = []
        best_neighbor_val = float('inf')
        # iterate over columns
        for i in range(len(current)):
            # Try to move queen to every other row in column
            for row in range(len(current)):
                # skip over non-move
                if row == current[i]:
                    continue
                new_state = current.copy()
                new_state[i] = row
                new_state_val = h(new_state)
                # if lower cost, replace best neighbor state
                if new_state_val < best_neighbor_val:
                    best_neighbor_state = new_state
                    best_neighbor_val = new_state_val
        # print(f'{best_neighbor_state}, cost; {best_neighbor_val}')
        # check if we found local minimum
        if best_neighbor_val >= current_val:
            return current, current_val, num_steps, init_h
        current = best_neighbor_state
        current_val = best_neighbor_val
        num_steps += 1
    return -1;

def random_restart_hill_climbing(max_restarts):
    """
    Chooses a new, initial state uniformly at random and begin steepest descent from this new initial state.
    This process will loop until a solution (no attacking queens) is found or until max restarts is reached
    :param max_restarts: an integer representing the maximum number of restarts to run hill climbing
    :return: Returns the final state, its cost, the number of restarts to find a solution,
    and the total number of steps to find a solution
    """
    steps_sum = 0
    for i in range(max_restarts):
        random_state = random.randint(0,7,7)
        state, val, steps, init_cost = hill_climbing(random_state)
        steps_sum += steps
        if val == 0:
            return state, val, i, steps_sum

def simulated_annealing(max_iterations):
    """
    Run simulated annealing with a temperature schedule T = 1 - t/100
    :param max_iterations:an integer representing the maximum number of restarts to run simulated annealing
    :return: Returns the final state and the minimum value encountered across all iterations
    """
    current = random.randint(0,7,7) # random initial state
    h_min = float('inf')
    for t in range(max_iterations):
        # update temperature
        temp = 1 - t/100
        if temp == 0:
            return current, h_min

        # random new successor state:
        next = current.copy()
        change_column = random.randint(7)
        new_row = random.randint(7)
        # make sure we generate a different state
        while(new_row == next[change_column]):
            new_row = random.randint(7)
        next[change_column] = new_row
        # update h*
        new_state_cost = h(next)
        if new_state_cost < h_min:
            h_min = new_state_cost

        delta_e = h(current) - new_state_cost
        if delta_e > 0:
            current = next
        else:
            prob = math.exp(delta_e / temp)
            if random.random() < prob:
                current = next
    return current, h_min

def genetic_algorithm(population_size, mutation_rate, max_iterations):
    """
    Run a genetic algorithm with the probability of an individual being chosen as a parent in a pair is proportional
    to its fitness, generating the initial population for each run uniformly at random
    :param population_size: Integer representing the number of individuals in the population
    :param mutation_rate: Float representing the chance of a child to mutate
    :param max_iterations: Integer representing the maximum number of iterations to update the population
    :return: Returns the minimum state cost encountered from all individuals
    """
    # Generate initial population
    population = []
    min_h = float('inf')
    for i in range(population_size):
        individual = random.randint(0, 7, 7)
        fitness = h(individual)
        if fitness < min_h: min_h = fitness
        population.append((individual, fitness))
    for iterations in range(max_iterations):
        population_2 = []
        for i in range(population_size):
            # fitnesses = [ind[1] for ind in population]
            fitnesses = [1.0 / (ind[1] + 1e-6) for ind in population]
            individuals = [ind[0] for ind in population]
            probabilities = np.array(fitnesses) / np.sum(fitnesses)
            parent_indices = random.choice(len(individuals), size=2, replace=False, p=probabilities)
            parent1, parent2 = population[parent_indices[0]], population[parent_indices[1]]

            child = reproduce(parent1, parent2)
            if random.random() < mutation_rate:
                child = mutate(child)
            population_2.append(child)
            if child[1] < min_h: min_h = child[1]
        population = population_2
    return min_h

def reproduce(parent1, parent2):
    """
    Takes two parent states and produces a child using a uniformly generated cross over point
    :param parent1: Tuple (array, cost) representing the first parent
    :param parent2: Tuple (array, cost) representing the first parent
    :return: Returns a tuple (array, cost) representing the offspring
    """
    n = len(parent1[0])
    c = random.randint(1, n)
    offspring = []
    for i in range(n):
        if i < c:
            offspring.append(parent1[0][i])
        else:
            offspring.append(parent2[0][i])
    # print(f'c value: {c}, p1: {parent1[0]}, p2: {parent2[0]}, offspring: {offspring}')
    fitness = h(offspring)
    return (offspring, fitness)
def mutate(individual):
    """
    Changes the row value of a random column in the individual's state
    :param individual: Tuple (array, cost) representing the child to mutate
    :return: Returns a tuple (array, cost) representing the mutated child
    """
    change_index = random.randint(0,7)
    new_val = random.randint(0,7)
    individual[0][change_index] = new_val
    individual[1] = h(individual[0])
    return individual

if __name__ == '__main__':
    # Hill Climbing
    # each number in state represents the row number of the queen in column c
    state = [4,2,4,0,4,1,5]
    hc_state, hc_val, hc_steps, hc_init_cost = hill_climbing(state)
    print(f'HILL CLIMBING: initial cost: {hc_init_cost}, final state: {hc_state}, final cost: {hc_val}, steps: {hc_steps}')

    # Random Restart Hill Climbing
    rrhc_state, rrhc_val, rrhc_restarts, rrhc_steps_sum = random_restart_hill_climbing(100)
    print(f'RANDOM RESTARTS: final state: {rrhc_state}, final cost: {rrhc_val}, number of restarts: {rrhc_restarts}')
    # run rrhc 1000 times:
    restarts_sum = 0
    steps_sum = 0
    for i in range(1000):
        rrhc_state, rrhc_val, rrhc_restarts, rrhc_steps_sum = random_restart_hill_climbing(100)
        restarts_sum += rrhc_restarts
        steps_sum += rrhc_steps_sum
    avg_restarts = restarts_sum / 1000.0
    avg_steps = steps_sum / 1000.0
    print(f'average number of restarts: {avg_restarts}, average total number of steps: {avg_steps}')

    # Simulated Annealing
    # run 10,000 times:
    min_cost_sum = 0
    for i in range(10000):
        sa_state, sa_min_cost = simulated_annealing(100)
        min_cost_sum += sa_min_cost
        # print(sa_min_cost)
    avg_min_cost = min_cost_sum / 10000.0
    print(f'SIMULATED ANNEALING: average min cost: {avg_min_cost}')

    # Genetic Algorithm
    min_h_sum = 0
    for i in range(10000):
        print(i)
        min_h = genetic_algorithm(4, 0.1, 100)
        min_h_sum += min_h
    avg_min_h = min_h_sum / 10000.0
    print(f'GENETIC ALGORITHM: average h*: {avg_min_h}')
