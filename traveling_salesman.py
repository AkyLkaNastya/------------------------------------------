import math
import random
import time
import numpy as np

def coordinate_calculator(x_1, y_1, x_2, y_2):
    x = x_2 - x_1
    y = y_2 - y_1
    return x, y

def fitness_function_EUC_2D(dist_1, dist_2):
    dist_1 = dist_1 ** 2
    dist_2 = dist_2 ** 2
    distance = math.sqrt(dist_1 + dist_2)
    return distance

def fitness_function_ATT(dist_1, dist_2):
    dist_1 = dist_1 ** 2
    dist_2 = dist_2 ** 2
    distance = math.ceil(math.sqrt((dist_1 + dist_2)/10))
    return distance

def fitness_function_MATRIX(i, j, distance_matrix):
    distance = distance_matrix[i][j]
    return distance

def route_generator(cities):
    route = list(cities.keys())
    random.shuffle(route)
    return route

def route_distance_calculator(route, cities_dict, weight_type, distance_matrix=None):
    total_route_distance = 0
    for i in range(len(route)):
        city_1 = route[i]
        city_2 = route[(i + 1) % len(route)]

        if weight_type == "EUC_2D" or weight_type == "ATT":
            x_1, y_1 = cities_dict[city_1]
            x_2, y_2 = cities_dict[city_2]
            x, y = coordinate_calculator(x_1, y_1, x_2, y_2)
            if weight_type == "EUC_2D":
                distance = fitness_function_EUC_2D(x, y)
            elif weight_type == "ATT":
                distance = fitness_function_ATT(x, y)
        elif weight_type == "EXPLICIT":
            distance = fitness_function_MATRIX(city_1-1, city_2-1, distance_matrix)
        else:
            raise ValueError(f"Неизвестный weight_type: {weight_type}")

        total_route_distance += distance
    return total_route_distance

def population_generator(size, cities_dict):
    return [route_generator(cities_dict) for _ in range(size)]

def roulette_wheel_selection(population, cities_dict, weight_type, distance_matrix=None):
    distances = [route_distance_calculator(route, cities_dict, weight_type, distance_matrix) for route in population]
    max_dist = max(distances)
    fitness = [max_dist - dist + 1 for dist in distances]
    total = sum(fitness)
    r = random.uniform(0, total)
    for i, f in enumerate(fitness):
        if r <= f:
            return population[i]
        r -= f
    return population[-1]

def partially_mapped_crossover(parent_1, parent_2):
    size = len(parent_1)
    start, end = sorted(random.sample(range(size), 2))

    child_1 = [None] * size
    child_2 = [None] * size

    child_1[start:end+1] = parent_2[start:end+1]
    child_2[start:end+1] = parent_1[start:end+1]

    mapping_1 = dict(zip(parent_2[start:end+1], parent_1[start:end+1]))
    mapping_2 = dict(zip(parent_1[start:end+1], parent_2[start:end+1]))

    for i in range(size):
        if i < start or i > end:
            element = parent_1[i]
            while element in mapping_1:
                element = mapping_1[element]
            child_1[i] = element

            element = parent_2[i]
            while element in mapping_2:
                element = mapping_2[element]
            child_2[i] = element

    return child_1, child_2

def mutation(individual):
    start, end = sorted(random.sample(range(len(individual)), 2))
    mutated = individual.copy()
    mutated[start:end+1] = reversed(mutated[start:end+1])
    return mutated

def genetic_algorythm_for_the_TSP(cities, weight_type, recommended_size, recommended_mutation_rate, distance_matrix=None, time_limit=1800):
    start_time = time.time()
    population_size = recommended_size
    generations = 1000
    mutation_rate = recommended_mutation_rate

    if weight_type in ("EUC_2D", "ATT"):
        cities_dict = {city[0]: (city[1], city[2]) for city in cities}
    elif weight_type == "EXPLICIT":
        cities_dict = {i+1: None for i in range(len(distance_matrix))}

    population = population_generator(population_size, cities_dict)

    def calculate_distance(individual):
        return route_distance_calculator(individual, cities_dict, weight_type, distance_matrix)

    best_individual = min(population, key=calculate_distance)
    best_distance = calculate_distance(best_individual)

    for generation in range(generations):
        new_population = []
        new_population.append(best_individual)

        for i in range (population_size):
            parent_1 = roulette_wheel_selection(population, cities_dict, weight_type, distance_matrix)
            parent_2 = roulette_wheel_selection(population, cities_dict, weight_type, distance_matrix)
            child_1, child_2 = partially_mapped_crossover(parent_1, parent_2)

            if np.random.random() < mutation_rate:
                child_1 = mutation(child_1)
            if np.random.random() < mutation_rate:
                child_2 = mutation(child_2)

            new_population.extend([child_1, child_2])

        population = new_population[:population_size]
        current_best = min(population, key=calculate_distance)
        current_distance = calculate_distance(current_best)

        if current_distance < best_distance:
            best_individual = current_best
            best_distance = current_distance

        if generation % 10 == 0:
            print(f"Поколение {generation}: Лучшая дистанция = {best_distance:.2f}")

        if time.time() - start_time > time_limit:
            print("Время вышло!")
            print(f"Поколение {generation}: Лучшая дистанция = {best_distance:.2f}")
            print(f"\nРазмер популяции: ", recommended_size)
            print(f"\nВероятность мутации: ", recommended_mutation_rate)
            break

    execution_time = time.time() - start_time
    return best_individual, best_distance, execution_time


with open('content/a280.tsp', 'r') as f:
    a280_cities = []
    for line in f:
        if line.strip().startswith("NODE_COORD_SECTION"):
            break
    for line in f:
        if line.strip() == "EOF":
            break
        parts = line.strip().split()
        a280_cities.append((int(parts[0]), float(parts[1]), float(parts[2])))
a280_data = {'dimension': 280, 'weight_type': 'EUC_2D', 'cities': a280_cities, 'param_size': 300, 'param_rate': 0.35}

with open('content/att48.tsp', 'r') as f:
    att48_cities = []
    for line in f:
        if line.strip().startswith("NODE_COORD_SECTION"):
            break
    for line in f:
        if line.strip() == "EOF":
            break
        parts = line.strip().split()
        att48_cities.append((int(parts[0]), float(parts[1]), float(parts[2])))
att48_data = {'dimension': 48, 'weight_type': 'ATT', 'cities': att48_cities, 'param_size': 60, 'param_rate': 0.15}

with open('content/bays29.tsp', 'r') as f:
    bays29_weights = []
    in_weight_section = False
    for line in f:
        if line.strip().startswith("EDGE_WEIGHT_SECTION"):
            in_weight_section = True
            continue
        if line.strip().startswith("DISPLAY_DATA_SECTION"):
            break
        if line.strip() == "EOF":
            break
        if in_weight_section:
            bays29_weights.extend(map(int, line.strip().split()))

bays29_matrix = [[0]*29 for _ in range(29)]
index = 0
for i in range(29):
    for j in range(i+1):
        if index < len(bays29_weights):
            bays29_matrix[i][j] = bays29_weights[index]
            bays29_matrix[j][i] = bays29_weights[index]
            index += 1
bays29_data = {'dimension': 29, 'weight_type': 'EXPLICIT', 'weight_matrix': bays29_matrix, 'param_size': 30, 'param_rate': 0.25}

with open('content/fl417.tsp', 'r') as f:
    fl417_cities = []
    for line in f:
        if line.strip().startswith("NODE_COORD_SECTION"):
            break
    for line in f:
        if line.strip() == "EOF":
            break
        parts = line.strip().split()
        fl417_cities.append((int(parts[0]), float(parts[1]), float(parts[2])))
fl417_data = {'dimension': 417, 'weight_type': 'EUC_2D', 'cities': fl417_cities, 'param_size': 450, 'param_rate': 0.13}

with open('content/ch150.tsp', 'r') as f:
    ch150_cities = []
    for line in f:
        if line.strip().startswith("NODE_COORD_SECTION"):
            break
    for line in f:
        if line.strip() == "EOF":
            break
        parts = line.strip().split()
        ch150_cities.append((int(parts[0]), float(parts[1]), float(parts[2])))
ch150_data = {'dimension': 150, 'weight_type': 'EUC_2D', 'cities': ch150_cities, 'param_size': 350, 'param_rate': 0.3}

with open('content/gr17.tsp', 'r') as f:
    gr17_weights = []
    for line in f:
        if line.strip().startswith("EDGE_WEIGHT_SECTION"):
            break
    for line in f:
        if line.strip() == "EOF":
            break
        gr17_weights.extend(map(int, line.strip().split()))

gr17_matrix = [[0]*17 for _ in range(17)]
index = 0
for i in range(17):
    for j in range(i+1):
        if index < len(gr17_weights):
            gr17_matrix[i][j] = gr17_weights[index]
            gr17_matrix[j][i] = gr17_weights[index]
            index += 1
gr17_data = {'dimension': 17, 'weight_type': 'EXPLICIT', 'weight_matrix': gr17_matrix, 'param_size': 30, 'param_rate': 0.25}
datasets = {
    "a280": a280_data,
    "att48": att48_data,
    "fl417": fl417_data,
    "ch150": ch150_data,
}
#(cities, weight_type, distance_matrix=None,  recommended_size, recommended_mutation_rate, time_limit=600)

for name, data in datasets.items():
    print(f"\n--- Обработка {name}.tsp ---")

    if data['weight_type'] == 'EXPLICIT':
        best_route, best_distance, execution_time = genetic_algorythm_for_the_TSP(
            cities=list(range(1, data['dimension'] + 1)),
            weight_type=data['weight_type'],
            distance_matrix=data['weight_matrix'],
            recommended_size = data['param_size'],
            recommended_mutation_rate = data['param_rate']
        )
    else:
        best_route, best_distance, execution_time = genetic_algorythm_for_the_TSP(
            cities=data['cities'],
            weight_type=data['weight_type'],
            recommended_size = data['param_size'],
            recommended_mutation_rate = data['param_rate']
        )

    print(f"Лучший маршрут ({name}.tsp):", best_route)
    print(f"Длина маршрута ({name}.tsp):", best_distance)
    print(f"Время выполнения ({name}.tsp):", execution_time/60, "минут")