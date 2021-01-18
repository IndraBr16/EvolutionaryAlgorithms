import matplotlib.pyplot as plt
from random import choice, randint, uniform, shuffle, sample


####################
# FITNESS FUNCTION #
####################
def extract_subgrids(sudoku_grid, i):
    """
    Extract the 9 elements from a (3 x 3) grid cell in a (9 x 9) Sudoku solution.
    :param sudoku_grid: The specified 9 x 9 Sudoku solution grid.
    :param i: The specified index of the column
    :return: Returns a list of 9 integers that are part of a specified Sudoku sub-grid.
    """
    return sudoku_grid[i:i + 3] + sudoku_grid[i + 9:i + 12] + sudoku_grid[i + 18:i + 21]


def extract_all_subgrids(sudoku_grid):
    """
    Divide a flat vector into vectors with 9 elements, representing 3 x 3 boxes in the corresponding 9 x 9 2D vector.
    These are the standard Sudoku boxes.
    :param sudoku_grid: The specified 9 x 9 Sudoku solution
    :return: Returns the list of 3 x 3 grids in a 9 x 9 Sudoku solution
    """
    return [extract_subgrids(sudoku_grid, i) for i in [0, 3, 6, 27, 30, 33, 54, 57, 60]]


def make_sudoku_2D(sudoku_grid):
    """
    Function will take a flat vector and make the row of 9 x 9 Sudoku grid 2-D.
    :param sudoku_grid: The specified 9 x 9 Sudoku solution
    :return: Returns a list of 81 integers, which removes the (9 x 9) dimension to make the Sudoku grid 2-D
    """
    return [sudoku_grid[i * 9:(i + 1) * 9] for i in range(9)]


def consistency(sudoku_grid):
    """
    Function will check how many different elements there are in each row. Ideally there should be 9 different elements,
    if there are no duplicates. This function can also be used for columns by applying the built-in zip() function
    as such: zip(*sudoku_grid).
    :param sudoku_grid: Check how many different elements there are in each row. Ideally there should be 9 different
    elements, if there are no duplicates.
    :return: Returns an integer value for the number of conflicts
    """
    conflict_score = 0
    visited = []

    # Iterate through every cell in every row in a 9x9 Sudoku grid.
    for row in sudoku_grid:
        for i in range(len(row)):
            # If cell value has not been visited, append its index into the visited list.
            if row[i] not in visited:
                visited.append(row[i])
            else:
                conflict_score += 1
        visited = []

    return conflict_score


def fitness_sudoku(candidate_solution):
    """
    Solution strings are evaluated by counting the number of duplicate symbols in rows or columns, creating a fitness
    score. Fewer duplicates presumably means a better solution string. This will be determined by that fitness score.
    :param candidate_solution: An individual candidate solution for a sudoku puzzle to be specified
    :return: Returns the total fitness score of the evaluated candidate solution for a sudoku puzzle.
    """

    # Checks consistency of the rows by giving fitness score based on the number of conflicts in each row.
    sudoku_solution = make_sudoku_2D(candidate_solution)
    fitness_score = consistency(sudoku_solution)

    # Checks consistency of columns by giving fitness score based on the number of conflicts in each column.
    fitness_score += consistency(zip(*sudoku_solution))

    # Checks consistency of sub-sudoku grids by giving fitness score based on number of conflicts in each (3 x 3) grid.
    fitness_score += consistency(extract_all_subgrids(candidate_solution))

    return fitness_score


###############################
# INDIVIDUAL -LEVEL OPERATORS #
###############################

# FUNCTIONS FOR GENERATING SUDOKU GRID SOLUTION SPACES
def generate_sudoku_grid(filename):
    """
    Function combines the following 3 functions below to easily generate a (9 x 9) Sudoku grid, where all numbers that
    were in the text file are filled and fixed into their corresponding positions, while all the “.” characters are
    replaced into the integer zeros, and later filled in with numbers ranging from 1 to 9.
    :param filename: The specified text file
    :return: List of lists of integers with all 0s replaced.
    """
    sudoku_list = divideList(generate_array(filename), 9)
    random_solution = replaceZeros(sudoku_list)
    return random_solution


def generate_array(filename):
    """
    Function generates a list of 81 integers by replacing any '.' with a 0, and subsequently ignoring any other
    character from a specified textfile that is not going to be part of the list.
    :param filename: The file path to be specified.
    :return: A list of 81 integers.
    """
    sudoku_list_of_numbers = []
    sudokuGrid = open(filename, "r")

    # For each line in the text file, strip() will get rid of white space at the beginning and end of the string, while
    # split() helps get a list, with no specified delimiter.
    for eachLine in sudokuGrid:
        eachLine.strip().split()

        for character in eachLine:
            if character == '.':
                sudoku_list_of_numbers.append(0)

            elif character == '-' or character == '\n' or character == '!':
                continue

            else:
                sudoku_list_of_numbers.append(int(character))

    return sudoku_list_of_numbers


def replaceZeros(created_grid):
    """
    Function replaces 0 with a random value from 1 to 9, essentially filling an unfilled Sudoku cell.
    :param created_grid: The list of lists (sudoku grid) to be specified
    :return: Returns the newly non-zero sudoku grid
    """
    candidate_grid = []

    # First for-loop cycles through every row of the created grid from the text file
    for every_row in created_grid:
        candidate_row = []

        # For each row, it will check whether the number in the given index is 0, meaning that Sudoku cell is unfilled.
        for number in range(0, 9):

            # If 0, attempt to replace that 0 with a random integer ranging from 1 to 9
            if every_row[number] == 0:
                rand_number = randint(1, 9)

                # While the random number exists in the current row or the new candidate row list, attempt to generate
                # another random value from 1 to 9 to replace old integer value of rand_number
                while rand_number in every_row or rand_number in candidate_row:
                    rand_number = randint(1, 9)

                # Append that number onto the currently-checked row
                candidate_row.append(rand_number)

            # Otherwise, if that value in the given index of that Sudoku row is anything but 0, append it.
            # This ensures that numbers from the text file that were initially hardcoded onto the file will remain in
            # the following sudoku row, unless it is a 0 (Represents ".").
            else:
                candidate_row.append(every_row[number])

        # After all numbers have been filled in the selected row, append it to a new list, which will create and form
        # the 9 x 9 sudoku grid.
        candidate_grid.append(candidate_row)

    return candidate_grid


def divideList(sudoku_list, number_of_sublists):
    """
    Function divides the list of 81 integers into 9 small sub-lists of integers.
    :param sudoku_list: The list of integers to be specified.
    :param number_of_sublists: The desired number of sub-lists in the list.
    :return: A list of sub-lists of integers.
    """
    number_of_sublists = max(1, number_of_sublists)
    list_of_lists = [sudoku_list[i:i + number_of_sublists] for i in range(0, len(sudoku_list), number_of_sublists)]
    return list_of_lists


# Crossover Operator for Chosen Representation
def crossover_individual(sudoku_grid1, sudoku_grid2):
    """
    Returns a list of randomly chosen rows from 2 sudoku grids. This is a uniform crossover, where bits are randomly
    copied from the first or from the second parent
    :param sudoku_grid1: The first sudoku grid, which will be random.
    :param sudoku_grid2: The second sudoku grid, which will be random.
    :return: A list of sudoku rows of 9 integers from different sudoku grids.
    """
    # Performs a uniform crossover by zipping 2 candidate solutions together and extracting a row from them, creating
    # a new breed.
    return [choice(row) for row in zip(sudoku_grid1, sudoku_grid2)]


# Mutation Operator for Chosen Representation
def mutate_individual(sudoku_grid, filename):
    """
    Function performs swapping of numbers in a random number of rows, depending on the fitness score of the candidate
    Sudoku solution grid. Number swapping will only happen on those that were originally represented as "." in the text
    file.
    :param sudoku_grid: The specified list of lists, representing a 9 x 9 sudoku grid.
    :param filename: The specified text file
    :return: The same sudoku grid with the specified rows swapped.
    """
    # The base_grid is a form of reference to properly select which numbers to swap without affecting the predetermined
    # cells. Within this function, the base_grid will be used to cross-check and also make use of the same text file
    # used to create the candidate solutions
    base_grid = divideList(generate_array(filename), 9)

    # This is a list to store all indexes of the possible sudoku cells that can be swapped with one another.
    rand_index = []
    randomness = uniform(0, 1)

    # rand_row = randint(0, 8)
    # rand_sub_row = randint(0, 2)

    # This selects the number of rows chosen to mutate depending on the fitness score of a candidate solution.
    # If fitness score / 5 is greater than 8, the minimum number of rows being selected will be 8.
    num_rows = min(8, fitness_sudoku(sudoku_grid) // 5)
    row_indexes = sample([x for x in range(9)], num_rows)

    for i in row_indexes:
        for number in range(len(base_grid[i])):

            # If the value found on the specified row of the base grid is 0, append that index into the rand_index list.
            if base_grid[i][number] == 0:
                rand_index.append(base_grid[i][number])

            elif base_grid[i][number] != 0:
                continue

        # If the length of the list of indexes is greater than 1, shuffle them and extract 2 indexes based on a pop().
        if len(rand_index) > 1 and randomness > MUTATION_RATE:
            shuffle(rand_index)
            rand1 = rand_index.pop()
            rand2 = rand_index.pop()

            # This is where 2 of the random indexes will be used to select sudoku cells on the same row and swap their
            # values.
            temp = sudoku_grid[i][rand1]
            sudoku_grid[i][rand1] = sudoku_grid[i][rand2]
            sudoku_grid[i][rand2] = temp

    # Pencil marking, checking whether

    return sudoku_grid


##############################
# POPULATION-LEVEL OPERATORS #
##############################
# Loops through the population size
def generate_population(filename, population_size):
    """
    Function takes specified file path to generate the solution spaces for a specified population size, which is
    essentially the number of possible sudoku solutions to be generated.
    :param filename: The specified text file.
    :param population_size: The desired population integer size.
    :return: Finding the sudoku solution spaces according to the population size (integer).
    """
    population_list = []

    # In this loop, for a number of iterations based on the popoulation size, every candidate solution generated will
    # be appended
    for each_solution in range(population_size):
        solution_space = generate_sudoku_grid(filename)
        population_list.append(solution_space)

    return population_list


def generate_fitness_scores(pop_list):
    """
    Function goes through a population size list of sudoku gird solutions, iterates through them and produces a fitness
    score for each of them, stored in a list.
    :param pop_list: The specified population size list of sudoku solution spaces.
    :return: A list of fitness scores based on the number of solution spaces specified in population list.
    """
    # Initialise a flat, 2-D list.
    flat_list = []
    fitness_scores = []

    # Iterate through each solution space
    for each_solution in pop_list:
        # Ensure that you append all values in one solution space into a flat, 2-D list.
        for each_row in each_solution:
            for number in each_row:
                flat_list.append(number)

        # Find the fitness score of that solution list of integers, then append and reset flat_list = []
        fitness_score = fitness_sudoku(flat_list)
        fitness_scores.append(fitness_score)
        flat_list = []

    return fitness_scores


def select_population(population_list, fitness_population):
    """
    Selects a specified percentage of the a list of candidate sudoku solutions by firstly sorting the population based
    on fitness scores on every candidate solution, and only taking a specified percentage of them to be used in the
    evolutionary algorithm.
    :param population_list: The list population_lists.
    :param fitness_population: The list fitness_scores.
    :return: A new list of selected individuals that will be used in the evolutionary algorithm.
    """
    sorted_population = sorted(zip(population_list, fitness_population), key=lambda ind_fit: ind_fit[1])

    # Returns a new list of selected individuals that will be used in the evolutionary algorithm based on the truncation
    # rate.
    return [ind_sudoku for ind_sudoku, fitness in sorted_population[0: int(POPULATION_SIZE * TRUNCATION_RATE)]]


def crossover_population(mating_pool_list):
    """
    Function uses crossover_individual() function to perform a uniform crossover between two parents to produce an
    offspring. This process creates a number of offsprings aimed to refill the population of the original mating pool
    back to the original population size, and then undergo crossover in the process of the evolutionary algorithm.
    This preserves the previously best-selected individuals from the initial population, and achieves population
    diversity.
    :param mating_pool_list: The list population_lists.
    :return: Returns a list of newly bred candidate Sudoku solutions including the best chosen ones from the
    initial truncation selection.
    """
    offspring_list = []

    # Creates the offsprings every iteration, and appends it onto an offspring list
    for crossovers in range(POPULATION_SIZE - len(mating_pool_list)):
        crossed_over_grid = crossover_individual(choice(mating_pool_list), choice(mating_pool_list))
        offspring_list.append(crossed_over_grid)

    # Return the mating pool along with the offsprings to create one population list that will all undergo crossover
    # in the later processes.
    return mating_pool_list + offspring_list


def mutate_population(population_list, filename):
    """
    Function uses and applies the mechanics of the individual-level mutation operator to perform mutations across all
    candidate Sudoku solution grids in a specified population.
    :param population_list: The list of sudoku solution grids.
    :param filename: The list fitness_scores for the corresponding Sudoku solution grids.
    :return: Returns a list of mutated sudoku solution grids
    """
    return [mutate_individual(ind_sudoku, filename) for ind_sudoku in population_list]


def best_population_of_sudokus(population_list, fitness_population):
    """
    Function that gets the best Sudoku solution grid along with its corresponding fitness score.
    :param population_list: The list of sudoku solution grids.
    :param fitness_population: The list fitness_scores for the corresponding Sudoku solution grids.
    :return: Returns a sorted
    """
    return sorted(zip(population_list, fitness_population), key=lambda individual_fitness: individual_fitness[1])[0]


##########################
# EVOLUTIONARY ALGORITHM #
##########################
def evolve(filename):
    """
    Function represents the evolutionary algorithm for solving Sudoku puzzles, implementing all problem-specific
    components to the task, for instance the appropriate solution space and solution representation, fitness function,
    crossover and mutation operators for the chosen representation, population initialisation, selection and
    replacement methods, and an appropriate termination criterion.
    :param filename: Specified name of file, which is essentially the Sudoku puzzle grid.
    :return: Returns a list of all the best fitness score in each generation. The size of the list will be dependent on
    the number generations that are request from user input.
    """
    population = generate_population(filename, POPULATION_SIZE)
    fitness_pop_scores = generate_fitness_scores(population)
    best_fitness_score = min(fitness_pop_scores)
    gen = 0
    total_gens = 0
    # This list will store all the best fitness score per generation.
    best_fits = []

    # While the total number of generations ran have not reached the specified value of NUMBER_GENERATIONS, keep running
    # the following code within the while loop.
    while total_gens < NUMBER_GENERATIONS:
        gen += 1
        total_gens += 1

        # Select population of individuals to undergo crossover and mutation
        mating_pool = select_population(population, fitness_pop_scores)

        # Perform crossover to create new offsprings
        offspring_population = crossover_population(mating_pool)

        # Mutate the crossed-over mating pool of offsprings
        population = mutate_population(offspring_population, filename)

        fitness_pop_scores = generate_fitness_scores(population)
        worst_fitness_score = max(fitness_pop_scores)
        best_sudoku_grid, best_fitness_score = best_population_of_sudokus(population, fitness_pop_scores)

        best_fits.append(best_fitness_score)

        # If gen counter reaches the reset point integer value, a technique that prevents convergence to local minima,
        # also known as Judgement Day, will keep the current candidate Sudoku solution with the best (lowest) fitness
        # score, and repopulate with new (population size - 1) candidate solutions.
        if gen == RESTART_POINT:
            print("#%2d" % total_gens, "Highest Fitness (Worst):%3d" % worst_fitness_score, "   ",
                  "Lowest Fitness (Best):%3d" % best_fitness_score)

            # When value equates to RESTART_POINT value, gen value will reset and continue to be incremented and
            # repeats the same process
            gen = 0

            selected_grid = best_sudoku_grid
            fitness_pop_scores.remove(best_fitness_score)
            population.remove(best_sudoku_grid)
            population = generate_population(filename, POPULATION_SIZE - 1)
            population.append(selected_grid)
            fitness_pop_scores = generate_fitness_scores(population)
            best_fitness_score = min(fitness_pop_scores)

            print("------------------------------------------RESTART-----------------------------------------------")
            print()

        # If the best fitness score reaches 0, print a statement indicating the success of the algorithm in finding a
        # solution to the given Sudoku puzzle. Subsequently, print the whole Sudoku Grid that presents the optimal
        # solution in a (9 x 9) manner. Print out the total number of generations ran to reach this point.
        elif best_fitness_score == 0:
            print()
            print("------------------------------------------FINISHED----------------------------------------------")
            for n in best_sudoku_grid:
                print(n)
            print()
            print("Awesome! Best Fit of 0 Reached!")
            print("Total number of Generations Ran: " + str(total_gens))
            print()
            break

    print(str(total_gens) + " Desired Generations Reached!")
    print()
    return best_fits


def plot_results(run1, run2, run3, run4, run5):
    """
    Function plots all 5 runs onto one graph.
    :param run1: First run of the experiment in evolve(filename)
    :param run2: Second run of the experiment in evolve(filename)
    :param run3: Third run of the experiment in evolve(filename)
    :param run4: Fourth run of the experiment in evolve(filename)
    :param run5: Fifth run of the experiment in evolve(filename)
    """
    # Plots 5 individual lines onto the same graph.
    plt.plot(run1, label="1st Run")
    plt.plot(run2, label="2nd Run")
    plt.plot(run3, label="3rd Run")
    plt.plot(run4, label="4th Run")
    plt.plot(run5, label="5th Run")

    plt.xlabel("Number of Generations")
    plt.ylabel("Fitness of the Best Candidate From Population")
    plt.title("Overall Evolutionary Algorithm Performance Graph For Population Size: {}".format(POPULATION_SIZE))
    plt.legend()
    plt.show()


def run_5_times(filename):
    """
    Function will iteratively run the same evolve() experiment 5 times for a specified filename as shown below.
    :param filename: Specified name of file, which is essentially the Sudoku puzzle grid.
    """
    total_runs = []
    for n in range(5):
        print("Performing Run #" + str(n + 1))
        total_runs.append(evolve(filename))

    plot_results(total_runs[0], total_runs[1], total_runs[2], total_runs[3], total_runs[4])


if __name__ == "__main__":
    grid1 = "grid1.txt"
    grid2 = "grid2.txt"
    grid3 = "grid3.txt"

    ##############
    # USER INPUT #
    ##############
    print("### WELCOME TO SUDOKU SOLVER USING AN EVOLUTIONARY ALGORITHM ###")
    print()
    choice_of_grid = input("Specify Grid File to Run [grid1, grid2 or grid3]: ")
    population_size_choice = input("Population Size [10, 100, 1000 or 10000]: ")
    POPULATION_SIZE = int(population_size_choice)
    truncation_choice = input("Truncation Rate [100% = 1.0; 50% = 0.5; and etc.]: ")
    TRUNCATION_RATE = float(truncation_choice)
    mutation_rate_choice = input("Mutation Rate Ranging from " + str(0.01) + " to " + str(1.00) + ": ")
    MUTATION_RATE = float(mutation_rate_choice)
    number_gens = input("Number of Generations to Be Done: ")
    NUMBER_GENERATIONS = int(number_gens)
    reset_point = input("Please Select a Reset Point After a Specified Number of Generations. \n" +
                        "This is to Prevent Convergence to a Local Minimum: ")
    RESTART_POINT = int(reset_point)
    print()
    print("Thank You For Your Input! The Program Will Run Shortly...")
    print()

    if choice_of_grid == "grid1":
        run_5_times(grid1)

    elif choice_of_grid == "grid2":
        run_5_times(grid2)

    elif choice_of_grid == "grid3":
        run_5_times(grid3)
