import numpy as np
import random
import operator
import sys


def fitness(grid, test_grid):
    assert len(grid) == len(test_grid)
    assert len(grid) and len(test_grid) and len(grid[0]) == len(test_grid[0])

    error = 0
    for row, row_test in zip(grid, test_grid):
        for i,j in zip(row, row_test):
            error += i^j

    return 1-error*1.0/(len(grid)*len(grid[0]))


def print_shape(arr):
    rows = arr.shape[0]
    cols = arr.shape[1]
    s = ''
    for x in range(rows):
        for y in range(cols):
            if arr[x, y]:
                s += '1'
            else:
                s += '0'
        print s
        s = ''


def generate_shape(x, y):
    temp = np.zeros((x,y), np.bool_)
    for i in range(x):
        for j in range(y):
            temp[i][j] = round(random.random())
    return temp


# print generate_shape(2,2)


# Shape is a tuple of (x,y) representing width and height
def generate_first_pop(pop_size, shape):
    population = []
    for i in range(pop_size):
        population.append(generate_shape(*shape))
    return population


def computePerfPopulation(population, desired):
    populationPerf = {}
    for idx, individual in enumerate(population):
        populationPerf[idx] = fitness(individual, desired)
    return sorted(populationPerf.items(), key = operator.itemgetter(1), reverse=True)


def selectFromPopulation(old_pop, populationSorted, best_sample, lucky_few):
    nextGeneration = []
    for i in range(best_sample):
        nextGeneration.append(old_pop[populationSorted[i][0]])
    for i in range(lucky_few):
        nextGeneration.append(old_pop[random.choice(populationSorted)[0]])
    random.shuffle(nextGeneration)
    return nextGeneration



def createChild(individual1, individual2):
    child = np.zeros(individual1.shape, np.bool_)
    for i in range(individual1.shape[0]):
        for j in range(individual2.shape[1]):
            if (int(100 * random.random()) < 50):
                child[(i,j)] += individual1[(i,j)]
            else:
                child[(i,j)] += individual2[(i,j)]
    return child


def createChildren(breeders, number_of_child):
    nextPopulation = []
    for i in range(len(breeders)):
        for j in range(number_of_child):
            nextPopulation.append(createChild(breeders[i], breeders[len(breeders) -1 -i]))
    return nextPopulation


def mutate_shape(arr):
    idx = (int(random.random() * arr.shape[0]),(int)(random.random() * arr.shape[1]))
    arr[idx] = random.random()


def mutatePopulation(population, chance_of_mutation):
    for i in range(len(population)):
        if random.random() < chance_of_mutation:
            mutate_shape(population[i])
    return population


def average_pop(population):
    avg = np.zeros(population[0].shape)
    for i in population:
        avg += i
    avg = avg/len(population)
    return avg


def error(population, actual):
    error = 0
    for i in population:
        error += 1-fitness(i,actual )
    return error


if __name__ == '__main__':
    x = np.array([[0, 0, 1,0,1,1,0], [1, 1, 1,0,0 , 1, 1]], np.bool_)
    y = np.array([[1, 0, 1], [1, 0, 0]], np.bool_)

    pop_size = 100
    mutation_rate = 0.05
    generations = 10


    print 'Goal:'
    print_shape(x)
    first = generate_first_pop(pop_size, x.shape)
    curr = first

    for i in range(generations):
        print 'Generation {}'.format(i)
        print average_pop(curr)
        print error(curr, x)
        sort_pop = computePerfPopulation(curr, x)

        choose_pop = selectFromPopulation(curr, sort_pop, int(pop_size//2.5) ,int(pop_size/2-pop_size/2.5))

        next_gen = createChildren(choose_pop, 2)

        curr = mutatePopulation(next_gen, mutation_rate)





