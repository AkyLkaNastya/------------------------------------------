import time
import random
import numpy as np
import pandas as pd

# c - вместимость
# p - ценность объектов
# s - оптимальный выбор объектов 
# w - веса объектов

# ============= ФУНКЦИИ ===============================================================

# Чтение файлов
def read_file_to_list(filename):
    with open(filename, 'r') as file:
        return [int(line.strip()) for line in file if line.strip()]
    
# Подсчёт веса и ценности рюкзака
def calculate_backpack(solution, weights, profits):
    total_weight = 0
    total_profit = 0
    
    for i in range(len(solution)):
        if solution[i] == 1:
            total_weight += weights[i]
            total_profit += profits[i]
    
    return total_weight, total_profit

# Уменьшаем вес
def delete_weight(solution, weights, profits, capacity):
    w, _ = calculate_backpack(solution, weights, profits)
    
    while w > capacity:
        ones_indices = [i for i, x in enumerate(solution) if x == 1]
    
        random_index = random.choice(ones_indices)
        solution[random_index] = 0
        
        w, _ = calculate_backpack(solution, weights, profits)

    return solution

# Рандомно заполняем массив нулями и единицами
def random_list(weights, profits, capacity):
    
    solution = [random.randint(0, 1) for _ in range(len(weights))]
    solution = delete_weight(solution, weights, profits, capacity)

    return solution

# Кроссовер
def crossover(a, b):
    n = len(a)
    part_size = round(n / 3)
    sizes = [part_size, part_size, n - 2 * part_size]
    
    a_parts = []
    b_parts = []
    start = 0
    for size in sizes:
        a_parts.append(a[start:start+size])
        b_parts.append(b[start:start+size])
        start += size
    
    a = a_parts[0] + b_parts[1] + a_parts[2]
    b = b_parts[0] + a_parts[1] + b_parts[2]
    
    return a, b

# Мутация
def flip_bit(solution):
    new_solution = solution.copy()
    indices = random.sample(range(len(solution)), 3)

    for index in indices:
        if 0 <= index < len(new_solution):
            new_solution[index] = 1 - new_solution[index]
    return new_solution

# Проверка лучшего результата и перезапись вероятностей
def best_things(solutions, weights, profits, max_profit, best_solution):
    sum_p = 0
    probabilities = []

    for _ in range(solutions_len):
        w, p = calculate_backpack(solutions[_], weights, profits)
        if p > max_profit:
            max_profit = p
            best_solution = solutions[_]
        sum_p += p
        probabilities.append(p)

    for _ in range(solutions_len):
        probabilities[_] = probabilities[_]/sum_p

    return max_profit, best_solution, probabilities

# ============= АЛГОРИТМ ===============================================================

def genetic_algorithm(solutions, capacity, profits, weights, probabilities):

    # Выбираем 2 подсписка на основе вероятностей
    selected_indices = np.random.choice(
        len(solutions), 
        size=2, 
        replace=False, 
        p=probabilities
    )

    selected = [solutions[i] for i in selected_indices]

    # Кроссоверим выбранные списки
    res_1, res_2 = crossover(selected[0], selected[1])

    # Добавляем мутацию flip bit к полученным решениям
    child_1 = flip_bit(res_1)
    child_2 = flip_bit(res_2)

    delete_weight(child_1, weights, profits, capacity)
    delete_weight(child_2, weights, profits, capacity)

    # Удаляем выбранные подсписки из исходного списка
    for i in sorted(selected_indices, reverse=True):
        del solutions[i]

    solutions.append(child_1)
    solutions.append(child_2)

    return solutions

# ============= ПРОХОД ПО ФАЙЛАМ ===============================================================

start = time.time()

# results_df = pd.read_csv('results.csv')
# results_list = []

files = 7
results = [[0,[],0,[]] for _ in range(files)] # Лучшие ценность/решение и оптимальные ценность/решение
    
for n in range(1, files+1):
    start_file = time.time()

    # Читаем файлы
    capacity = read_file_to_list(f'benchmarks/P0{n}/p0{n}_c.txt')[0]
    profits = read_file_to_list(f'benchmarks/P0{n}/p0{n}_p.txt')
    weights = read_file_to_list(f'benchmarks/P0{n}/p0{n}_w.txt')
    opt_selection = read_file_to_list(f'benchmarks/P0{n}/p0{n}_s.txt')

    _, p = calculate_backpack(opt_selection, weights, profits)
    results[n-1][2] = p
    results[n-1][3] = opt_selection

    solutions = [[],[],[],[],[]]
    solutions_len = len(solutions)

    for i in range(5):
        solutions[i] = random_list(weights, profits, capacity)

    best_solution = []
    max_profit = 0

    max_profit, best_solution, probabilities = best_things(solutions, weights, profits, max_profit, best_solution)

    # Триггер для остановки алгоритма. Если после 15 прогонов не будет результата лучше, то останавливаем цикл
    count = 0

    while count < 15:
        solutions = genetic_algorithm(solutions, capacity, profits, weights, probabilities)
        new_max_profit, best_solution, probabilities = best_things(solutions, weights, profits, max_profit, best_solution)
        
        if new_max_profit == max_profit:
            count += 1
        else:
            count = 0
            max_profit = new_max_profit

    results[n-1][0] = max_profit
    results[n-1][1] = best_solution

    end_file = time.time()

    print(f'==== #P0{n} ============== {end_file-start_file:.6} сек')
    print(f'Лучшая ценность:      {results[n-1][0]}')
    print(f'Лучшее решение:       {results[n-1][1]}')
    print()
    print(f'Оптимальная ценность: {results[n-1][2]}')
    print(f'Оптимальное решение:  {results[n-1][3]}')
    print()

    # Код для записи в DataFrame
    # 
    # current_time = end_file - start_file

    # existing_row = results_df[results_df['benchmark'] == f'P0{n}']
    
    # if not existing_row.empty:
    #     existing_index = existing_row.index[0]
    #     existing_profit = existing_row['Best profit'].values[0]
        
    #     if max_profit > existing_profit:
    #         # Обновляем только если новый результат лучше
    #         results_df.at[existing_index, 'Best profit'] = max_profit
    #         results_df.at[existing_index, 'Best solution'] = str(best_solution)
    #         # Вычисляем среднее время
    #         old_time = existing_row['time'].values[0]
    #         new_time = (old_time + current_time) / 2
    #         results_df.at[existing_index, 'time'] = round(new_time, 6)
    # else:
    #     new_row = {
    #         'benchmark': f'P0{n}',
    #         'time': round(current_time, 6),
    #         'Best profit': max_profit,
    #         'Best solution': str(best_solution),
    #         'Opt profit': results[n-1][2],
    #         'Opt solution': str(results[n-1][3])
    #     }
    #     results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

end = time.time()


# ============= ВЫВОДЫ ===============================================================

print(f'Время работы алгоритма: {(end-start):.6f}\n')