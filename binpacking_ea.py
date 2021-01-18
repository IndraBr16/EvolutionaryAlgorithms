import sys
import matplotlib.pyplot as plt
import pandas as pd
from random import choice, randint, seed

# For any Python packages not installed as according to requirements.txt, do open the file to check if they are
# already installed. Otherwise, type "python -m pip install -r requirements.txt" in Command Prompt.


##############################
# POPULATION-LEVEL OPERATORS #
##############################
def generate_candidate(num_bins, num_items):
    """
    Function that generates a candidate solution list.
    :param num_bins: Number of bins.
    :param num_items: Number of items (Default = 500).
    :return: Returns a candidate solution of how the items will be sorted amongst b bins.
    """
    candidate = []

    for i in range(num_items):
        candidate.append(randint(1, num_bins))

    return candidate


def find_worst(fitness_scores):
    """
    Helper function for weakest_replacement() to find the worst fitness score and its index
    in a list of candidates by iterating through a list of fitness scores.
    :param fitness_scores: List of fitness scores of candidates.
    :return: The worst fitness score in integers (worst), and the index (worst_idx) of that fitness
             score in the list, which corresponds to the score's candidate in the population list.
    """
    # Considering the fitness score of 0.
    worst = -1
    worst_idx = None
    for i in range(len(fitness_scores)):
        if worst < fitness_scores[i]:
            worst = fitness_scores[i]
            worst_idx = i

        elif worst == fitness_scores[i]:
            worst_idx = choice([worst_idx, i])
            worst = fitness_scores[worst_idx]

    return worst, worst_idx


# REPLACEMENT OPERATOR
def weakest_replacement(new_solution, new_sol_fitness, population_list, fitness_list):
    """
    Function that overwrites the worst candidate solution in the population with the new crossed-over and mutated
    child solution.
    :param new_solution: The newly crossed-over and mutated child solution for BPP.
    :param new_sol_fitness: The fitness score of the newly crossed-over and mutated child solution for BPP.
    :param population_list: List of candidate solutions.
    :param fitness_list: List of fitness scores for each candidate solution in population_list.
    :returns: Updated population list of candidate solutions, and the updated list of their respective fitness scores.
    """
    worst_fitness, worst_idx = find_worst(fitness_list)

    # If new solution fitness is better...
    if new_sol_fitness < worst_fitness:
        population_list[worst_idx] = new_solution
        fitness_list[worst_idx] = new_sol_fitness

    # If fitness scores are the same, break ties randomly.
    elif new_sol_fitness == worst_fitness:

        # If randint is 1, replace, else don't.
        if randint(1, 2) == 1:
            population_list[worst_idx] = new_solution
            fitness_list[worst_idx] = new_sol_fitness

    return population_list, fitness_list


def find_best(fitness_scores):
    """
    Helper function, for tournament selection, to find the best fitness scores amongst the selected chromosomes.
    :param fitness_scores: List of fitness scores for each candidate solution in population_list.
    :return: Returns the index of the best fitness score in the given fitness_scores list.
    """
    # Considering fitness score of 0.
    best = sys.maxsize
    best_idx = None

    for i in range(len(fitness_scores)):

        if best > fitness_scores[i]:
            best = fitness_scores[i]
            best_idx = i

        # If fitness scores are the same between two chromosomes being looked at, break ties randomly.
        elif best == fitness_scores[i]:
            best_idx = choice([best_idx, i])
            best = fitness_scores[best_idx]

    return best_idx


# POPULATION SELECTION OPERATOR
def tournament_selection(population_list, fitness_list, tournament_size):
    """
    Function that returns the best fit chromosome among selected number of chromosomes. When a sampling unit is drawn
    from a finite population and is returned to that population, after its characteristic(s) have been recorded,
    before the next unit is drawn, the sampling is said to be “with replacement”.
    :param population_list: List of candidate solutions.
    :param fitness_list: List of fitness scores for each candidate solution in population_list.
    :param tournament_size: Number of chromosomes to be selected. This argument will give flexibility towards performing
                            further experimentation on varying tournament sizes without requiring to make 2 functions -
                            one for binary tournament selection, and another that can perform with any tournament size.
    :return: Returns the fittest chromosome (candidate solution) among the selected candidates (For the 6 experiments
             done with BPP1 and BPP2, excluding further experiments tournament size will be 2, so function returns the
             fittest chromosome among the 2 selected.
    """
    # Create lists to store chromosomes and their corresponding fitness scores.
    chromosomes = []
    fitness_scores = []

    # Randomly choose a chromosome from the population; call it a, and another chromosome from the population;
    # call this b, and so on...
    for i in range(tournament_size):
        index = randint(0, len(population_list) - 1)
        chromosomes.append(population_list[index])
        fitness_scores.append(fitness_list[index])

    best_idx = find_best(fitness_scores)
    best_chromosome = chromosomes[best_idx]

    return best_chromosome


####################################
# CROSSOVER AND MUTATION OPERATORS #
####################################
def single_point_crossover(parent_a, parent_b):
    """
    Function that performs single-point crossover on 2 parents and returns 2 new children solutions.
    :param parent_a: A chromosome from the population, called a.
    :param parent_b: A chromosome from the population, called b.
    :return: Two new children solutions, c and d.
    """
    # Randomly select a ‘crossover point’, which should be smaller than the total length of the chromosome.
    # This will be ensured by having a "+ 1" at the end of the randint function, such that [0:] would be [1:]
    # to ensure that the crossover point is not the whole chromosome.
    idx = randint(0, len(parent_a) - 1) + 1
    # print(idx)

    # With "+ 1" implemented in idx variable above, parent_a[0:idx] or parent[0:idx] will not return [] when
    # randint value is 0.
    c = parent_a[0:idx] + parent_b[idx:]
    d = parent_b[0:idx] + parent_a[idx:]

    return c, d


def multi_gene_mutation(parent, k, num_bins):
    """
    Function that uses the parameterised mutation operator, Mk, where k is an integer.
    The Mk operator repeats the following k times: choose a gene (locus) at random, and change it to a random new
    value within the permissible range (num_bins). M1 changes one gene, while M15 could change 15 or fewer genes,
    because there is a chance that same gene might be chosen more than once.
    :param parent: Chromosome created from single-point crossover.
    :param k: Mutation operator for this, Mk, where k is an integer.
    :param num_bins: The number of bins becoming the permissible range to consider when mutating a gene.
    :return: Returns the new mutated candidate as the variable "parent".
    """
    for iteration in range(k):
        # Select a random gene by retrieving a random index ranging from 0 to the length of the list
        gene = randint(0, len(parent) - 1)

        # Replace the old gene with new gene.
        parent[gene] = randint(1, num_bins)

    return parent


####################
# FITNESS FUNCTION #
####################
def fitness(candidate, item_weights):
    """
    Function that returns the fitness score of the solution, by calculating the difference, d,
    between the heaviest and lightest bins.
    :param candidate: A candidate solution for BPP.
    :param item_weights: The weights of all items in candidate solution.
    :return: Difference value between the 2 given bin weights in integer form.
    """
    # Item number is not necessary here.
    bins_dict = {}

    # Iterate through every item and its corresponding weight.
    for bin_item, weight in zip(candidate, item_weights):

        # As bin_item represents the bin that item is in, if bin_item exists in the dictionary as a key
        if bin_item in bins_dict:
            bins_dict[bin_item] += weight

        else:
            bins_dict[bin_item] = weight

    bin_weights = bins_dict.values()
    # print(bins_dict.values())
    # print(max(bin_weights), min(bin_weights))

    return max(bin_weights) - min(bin_weights)


##########################
# EVOLUTIONARY ALGORITHM #
##########################
def ea_bpp(population_size, tournament_size, num_bins, num_items, k, cross_over=True, mutation=True):
    """
    Function represents the evolutionary algorithm for solving the Bin Packing Problem (BPP), implementing all
    problem-specific components to the task, for instance the appropriate solution space and solution representation,
    fitness function, crossover and mutation operators for the chosen representation, population initialisation,
    selection and replacement methods, and an appropriate termination criterion.
    :param population_size: The population size of the randomly generated solutions.
    :param tournament_size: The number of candidates to be selected from population. This value will tie in with the
                            tournament_selection() function, which requires it. Reason for this design of requiring a
                            tournament size value is to provide more flexibility for further experimentation without
                            inefficiently constructing similar functions.
    :param num_bins: The number of bins (b=10 if bpp_type is bpp1, and b=100 if bpp_type is bpp2)
    :param num_items: The number of bin items (n=500 by default).
    :param k: Mutation operator, Mk, where k is an integer.
    :param cross_over: Boolean to determine if crossover operation is being used.
    :param mutation: Boolean to determine if mutation operation is being used.
    :return: Returns the best solution, its best fitness score, and plotting data for the EA run
             (will be used for line charts).
    """
    # candidate_pop is a list used store the population of candidate solutions to BPP problem.
    candidate_pop = []

    # fitness_scores is a list used store the fitness scores of corresponding candidate solutions in candidate_pop list.
    fitness_scores = []

    # Counter used to keep track of the number of fitness evaluations done.
    fitness_evaluations = 0

    # plotting_data is used to store the best fitness scores obtained in each evaluation of 10000 fitness evaluations,
    # which is intended for line chart plotting purposes in other functions.
    plotting_data = []

    for i in range(population_size):
        candidate = generate_candidate(num_bins=num_bins, num_items=num_items)
        candidate_pop.append(candidate)

    # Weights will not change regardless of any solution generated by the function above.
    weights = []

    # If number of bins is 10, perform BPP1 weighting for all 500 items.
    if num_bins == 10:
        for i in range(num_items):
            weights.append((i + 1) * 2)

    # If number of bins is 100, perform BPP2 weighting for all 500 items.
    elif num_bins == 100:
        for i in range(num_items):
            weights.append(((i + 1) ** 2)*2)

    # Calculate fitness scores
    for candidate in candidate_pop:
        fitness_scores.append(fitness(candidate, item_weights=weights))
    # print(candidate_pop)
    print("Initial Best Fitness: " + str(min(fitness_scores)))

    # Keep recursively running the algorithm while fitness_evaluations value has not reached 10000
    # fitness_evaluations starts from 0 to 9999.
    while fitness_evaluations < 10000:

        # Use binary tournament selection (with replacement) twice to select two parents a and b.
        a = tournament_selection(population_list=candidate_pop,
                                 fitness_list=fitness_scores,
                                 tournament_size=tournament_size)

        b = tournament_selection(population_list=candidate_pop,
                                 fitness_list=fitness_scores,
                                 tournament_size=tournament_size)

        # Declare variables e and f as the end result from performing either one of the 3
        # mutation and crossover conditions.
        e = None
        f = None
        fitness_e = None
        fitness_f = None

        # cross_over boolean is False, so run the evolutionary algorithm without single-point crossover.
        if cross_over is False and mutation is True:

            # Run mutation on c and d to give two new solutions e and f. Evaluate the fitness of e and f.
            e = multi_gene_mutation(a, k, num_bins)
            fitness_e = fitness(e, item_weights=weights)

            f = multi_gene_mutation(b, k, num_bins)
            fitness_f = fitness(f, item_weights=weights)

        # mutation boolean is False, so run evolutionary algorithm without mutation.
        elif cross_over is True and mutation is False:

            # Run single-point crossover to return two children solutions, being e and f, which are the variables
            # declared above the IF and ELIF statements.
            e, f = single_point_crossover(a, b)

            # Evaluate the fitness of e and f.
            fitness_e = fitness(e, item_weights=weights)
            fitness_f = fitness(f, item_weights=weights)

        # Run evolutionary algorithm using both single-point crossover and multi-gene mutation.
        elif cross_over is True and mutation is True:
            # Run single-point crossover on these parents to give 2 children, c and d
            c, d = single_point_crossover(a, b)

            # Run mutation on c and d to give two new solutions e and f. Evaluate the fitness of e and f.
            e = multi_gene_mutation(c, k, num_bins)
            fitness_e = fitness(e, item_weights=weights)

            f = multi_gene_mutation(d, k, num_bins)
            fitness_f = fitness(f, item_weights=weights)

        # Run first weakest replacement for e.
        candidate_pop, fitness_scores = weakest_replacement(e, new_sol_fitness=fitness_e,
                                                            population_list=candidate_pop,
                                                            fitness_list=fitness_scores)

        # Run second weakest replacement for f.
        candidate_pop, fitness_scores = weakest_replacement(f, new_sol_fitness=fitness_f,
                                                            population_list=candidate_pop,
                                                            fitness_list=fitness_scores)

        fitness_evaluations += 1

        plotting_data.append(min(fitness_scores))

    print(fitness_evaluations)
    final_pop_fitness_tup = zip(candidate_pop, fitness_scores)
    best_solution = min(final_pop_fitness_tup, key=lambda t: t[1])

    # best_solution[0] is the best candidate solution obtained after 10000 fitness evaluations.
    # best_solution[1] is the best candidate's corresponding fitness score.
    # plotting_data is a list of the best fitness scores obtained in each evaluation of 10000 fitness evaluations,
    # intended for line chart plotting purposes.
    return best_solution[0], best_solution[1], plotting_data


####################################
# Run Experiments and Plot Results #
####################################
def run_experiment(trials, b, mk, pop_size, tour_size, num_items, experiment_no, bpp, cross_over=True, mutation=True):
    """
    Function used to run a specific experiment for BPP1 or BPP2 with a specific number of trials and other parameters.
    :param trials: Number of trials to be run.
    :param b: Number of bins.
    :param mk: Mutation operator, Mk, where k is an integer.
    :param pop_size: Population size of candidate solutions.
    :param tour_size: The number of candidates to be selected from population. This value will tie in with the
                      tournament_selection() function, which requires it. Reason for this design of requiring a
                      tournament size value is to provide more flexibility for further experimentation without
                      inefficiently constructing similar functions.
    :param num_items: Number of bin items.
    :param experiment_no: A string to show the Experiment number based on implementation instructions given.
    :param bpp: A string determining whether it is BPP1 or BPP2 for plotting title.
    :param cross_over: Boolean to determine if crossover operation is being used.
    :param mutation: Boolean to determine if mutation operation is being used.
    :return: Returns a list of best fitness scores from all 5 trials the EA ran, which will be used in
             run_all_bpp_experiments() function to plot table of results for each experiment's 5 trials and average
             fitness score computed from those trials.
    """
    # Create a list of different seed values to be used for each of the 5 trials. Values are fixed from here to ensure
    # reproducibility of results. Random seed generation will use values 1, 2, 3, 4 and 5.
    random_state = [1, 2, 3, 4, 5]

    # Create a list to store the 5 best solutions for 5 trials.
    best_solutions = []

    # Create a list to store the 5 best fitness scores in correspondence to the 5 best solutions for 5 trials.
    best_fitness_scores = []

    # Store a list of data points for all trials.
    line_graph_data = []

    for i in range(trials):
        # Seed the run with different random number seeds.
        seed(random_state[i])

        print("Trial #" + str(i+1) + ":")
        print("Seed Value is {}".format(random_state[i]))
        solution, sol_fitness, plot_data = ea_bpp(population_size=pop_size,
                                                  tournament_size=tour_size,
                                                  num_bins=b,
                                                  num_items=num_items,
                                                  k=mk,
                                                  cross_over=cross_over,
                                                  mutation=mutation)

        # Save best solution at the end of 10000 fitness evaluations into best_solutions.
        best_solutions.append(solution)

        # Save best solution's fitness score at the end of 10000 fitness evaluations into best_fitness_scores.
        best_fitness_scores.append(sol_fitness)

        # Save plotting data for each trial into line_graph_data.
        line_graph_data.append(plot_data)

        print("Best Solution is: " + str(solution) + "\nFinal Fitness Score: " + str(sol_fitness))
        print()

    # Create plotting labels for each line graph drawn in the line chart.
    labels = [str(i+1) for i in range(trials)]
    plt.figure(figsize=(10, 6))

    # for-loop iteratively plots all 5 line graphs in one chart.
    for each_trial, each_label in zip(line_graph_data, labels):

        plt.plot(each_trial, label="Trial #" + str(each_label))

    plt.xlabel("Fitness Evaluations Iteration (10000)")
    plt.ylabel("Fitness Score")
    plt.title("{} Experiment {} \n(Bins = {}, Pop. Size = {}, Mk = {}, Crossover = {}, Mutation = {})"
              .format(bpp,
                      str(experiment_no),
                      b,
                      pop_size,
                      mk,
                      cross_over,
                      mutation))
    plt.legend()
    plt.show()

    return best_fitness_scores


def plot_table_results(data_set, average_fitness, bpp):
    """
    Function that plots a table of results of the best solution fitness scores attained in each trial for the 6
    experiments performed, and also plotting a bar chart in aid of visualizing the average fitness scores for each
    trial done.
    :param data_set: The dataset used to plot the table.
    :param average_fitness: The list of average fitness scores attained for each experiment.
    :param bpp: The BPP problem name, in String format, to add to the plotting title.
    """
    cell_text = []

    row_labels = ["Exp #" + str(i + 1) for i in range(len(average_fitness))]
    print(row_labels)
    col_labels = ["Trial #" + str(i + 1) for i in range(5)]
    col_labels.append("Average")
    table_df = pd.DataFrame(data_set, columns=col_labels)
    print(table_df)

    # Iterate through dataframe and append all values into the list
    for row in range(len(table_df)):
        cell_text.append(table_df.iloc[row])

    # Plot results on a table.
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].axis('tight')
    axs[0].axis('off')

    the_table = axs[0].table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels, loc='center')
    the_table.set_fontsize(10)

    axs[1].bar(["1", "2", "3", "4", "5", "6"], average_fitness)
    axs[1].yaxis.grid(True, linestyle='--', which='major',
                      color='grey', alpha=.25)

    plt.title("Evaluation of Fitness Scores for Each Experiment (" + bpp + ")")
    plt.xlabel("Experiment Number #_")
    plt.ylabel("Fitness Score")
    plt.show()


def run_all_bpp_experiments(bpp, num_bins):
    """
    Function that runs the specified 6 experiments requested in CA specifications.
    :param bpp: The BPP problem name, in String format, to add to the plotting title.
    :param num_bins: Number of bins.
    """
    # A list to store a list of best solution fitness scores from each experiment
    bpp_table = []

    # A list to store average fitness scores of all 6 experiments.
    ave_fitness_scores = []

    print("{} Experiment #1"
          "\n(Number of Bins = {}; Population Size = 10; Mk = 1; Crossover = True) ".format(bpp,
                                                                                            str(num_bins)))
    exp1_fitness_scores = run_experiment(trials=5,
                                         b=num_bins,
                                         mk=1,
                                         pop_size=10,
                                         tour_size=2,
                                         num_items=num_bin_items,
                                         experiment_no=1,
                                         bpp=bpp,
                                         cross_over=True,
                                         mutation=True)

    # Take average fitness score from all 5 trials
    average_fitness1 = sum(exp1_fitness_scores) / len(exp1_fitness_scores)
    ave_fitness_scores.append(average_fitness1)

    # Append that into list of fitness scores for table plotting purposes
    exp1_fitness_scores.append(average_fitness1)
    bpp_table.append(exp1_fitness_scores)
    print()

    print("{} Experiment #2"
          "\n(Number of Bins = {}; Population Size = 100; Mk = 1; Crossover = True) ".format(bpp,
                                                                                             str(num_bins)))
    exp2_fitness_scores = run_experiment(trials=5,
                                         b=num_bins,
                                         mk=1,
                                         pop_size=100,
                                         tour_size=2,
                                         num_items=num_bin_items,
                                         experiment_no=2,
                                         bpp=bpp,
                                         cross_over=True,
                                         mutation=True)

    # Take average fitness score from all 5 trials
    average_fitness2 = sum(exp2_fitness_scores) / len(exp2_fitness_scores)
    ave_fitness_scores.append(average_fitness2)

    # Append that into list of fitness scores for table plotting purposes
    exp2_fitness_scores.append(average_fitness2)
    bpp_table.append(exp2_fitness_scores)
    print()

    print("{} Experiment #3"
          "\n(Number of Bins = {}; Population Size = 10; Mk = 5; Crossover = True) ".format(bpp,
                                                                                            str(num_bins)))
    exp3_fitness_scores = run_experiment(trials=5,
                                         b=num_bins,
                                         mk=5,
                                         pop_size=10,
                                         tour_size=2,
                                         num_items=num_bin_items,
                                         experiment_no=3,
                                         bpp=bpp,
                                         cross_over=True,
                                         mutation=True)

    # Take average fitness score from all 5 trials
    average_fitness3 = sum(exp3_fitness_scores) / len(exp3_fitness_scores)
    ave_fitness_scores.append(average_fitness3)

    # Append that into list of fitness scores for table plotting purposes
    exp3_fitness_scores.append(average_fitness3)
    bpp_table.append(exp3_fitness_scores)
    print()

    print("{} Experiment #4"
          "\n(Number of Bins = {}; Population Size = 100; Mk = 5; Crossover = True) ".format(bpp,
                                                                                             str(num_bins)))
    exp4_fitness_scores = run_experiment(trials=5,
                                         b=num_bins,
                                         mk=5,
                                         pop_size=100,
                                         tour_size=2,
                                         num_items=num_bin_items,
                                         experiment_no=4,
                                         bpp=bpp,
                                         cross_over=True,
                                         mutation=True)

    # Take average fitness score from all 5 trials
    average_fitness4 = sum(exp4_fitness_scores) / len(exp4_fitness_scores)
    ave_fitness_scores.append(average_fitness4)

    # Append that into list of fitness scores for table plotting purposes
    exp4_fitness_scores.append(average_fitness4)
    bpp_table.append(exp4_fitness_scores)
    print()

    print("{} Experiment #5"
          "\n(Number of Bins = {}; Population Size = 10; Mk = 5; Crossover = False) ".format(bpp,
                                                                                             str(num_bins)))
    exp5_fitness_scores = run_experiment(trials=5,
                                         b=num_bins,
                                         mk=5,
                                         pop_size=10,
                                         tour_size=2,
                                         num_items=num_bin_items,
                                         experiment_no=5,
                                         bpp=bpp,
                                         cross_over=False,
                                         mutation=True)

    # Take average fitness score from all 5 trials
    average_fitness5 = sum(exp5_fitness_scores) / len(exp5_fitness_scores)
    ave_fitness_scores.append(average_fitness5)

    # Append that into list of fitness scores for table plotting purposes
    exp5_fitness_scores.append(average_fitness5)
    bpp_table.append(exp5_fitness_scores)
    print()

    print("{} Experiment #6"
          "\n(Number of Bins = {}; Population Size = 10; Mk = 5; Crossover = False) ".format(bpp,
                                                                                             str(num_bins)))
    exp6_fitness_scores = run_experiment(trials=5,
                                         b=num_bins,
                                         mk=0,
                                         pop_size=10,
                                         tour_size=2,
                                         num_items=num_bin_items,
                                         experiment_no=6,
                                         bpp=bpp,
                                         cross_over=True,
                                         mutation=False)

    # Take average fitness score from all 5 trials
    average_fitness6 = sum(exp6_fitness_scores) / len(exp6_fitness_scores)
    ave_fitness_scores.append(average_fitness6)

    # Append that into list of fitness scores for table plotting purposes
    exp6_fitness_scores.append(average_fitness6)
    bpp_table.append(exp6_fitness_scores)
    print()

    # Function below plots the table of results using bpp_table, and a bar chart using ave_fitness_scores, while
    # bpp is a string used to name the bar chart's title.
    plot_table_results(bpp_table, ave_fitness_scores, bpp)


#######################
# Further Experiments #
#######################

# Analyzing effects of changes in tournament size of selection method used.
def varying_tour_size_exp(bpp, num_bins, trials):
    """
    Function that performs an experiment on varying tournament sizes, not just considering binary tournament selection.
    :param bpp: The BPP problem name, in String format, to add to the plotting title.
    :param num_bins: Number of bins.
    :param trials: Number of trials.
    """
    # Create a list to store fitness scores from each tournament size implemented on the EA.
    tournament_results = []

    # Create a list to data in a table format.
    cell_text = []

    # Tournament size should start from 2, as binary being the simplest form of tournament selection, then progressing
    # to higher tournament sizes. To ensure that results are consistent, the only changing variable will be the
    # tournament size, nothing more.
    tournament_sizes = [2, 3, 4, 5, 6]

    # Random seed generation will use values 1, 2, 3, 4 and 5.
    random_state = [1, 2, 3, 4, 5]

    # Create a list to store the 5 best fitness scores in correspondence to the 5 best solutions for 5 trials.
    best_fitness_scores = []

    for tour_size in tournament_sizes:
        for i in range(trials):
            seed(random_state[i])

            print("Trial #" + str(i + 1) + ":")
            print("Seed Value is {}".format(random_state[i]))
            solution, sol_fitness, plot_data = ea_bpp(population_size=10,
                                                      tournament_size=tour_size,
                                                      num_bins=num_bins,
                                                      num_items=500,
                                                      k=1,
                                                      cross_over=True,
                                                      mutation=True)
            print("Final Fitness Score: " + str(sol_fitness))
            print()

            best_fitness_scores.append(sol_fitness)

        average_fitness = sum(best_fitness_scores) / len(best_fitness_scores)
        print("Average Fitness For Tournament Size {}: {}".format(tour_size, average_fitness))
        print()
        print()
        tournament_results.append(round(average_fitness, 2))

    row_labels = [str(bpp) + " Tournament Size " + str(size) for size in tournament_sizes]
    table_df = pd.DataFrame(tournament_results, columns=["Average Fitness Scores"])

    # Iterate through dataframe and append all values into the list
    for each_row in range(len(table_df)):
        cell_text.append(table_df.iloc[each_row])

    # Plot results on a table.
    plt.table(cellText=cell_text,
              rowLabels=row_labels,
              colLabels=["Average Fitness Scores"],
              colWidths=[0.3],
              loc='center')
    plt.axis('tight')
    plt.axis('off')
    plt.show()


# Analyzing effects of changes in mutation rate, Mk (change in k value, where k is an integer).
def varying_mutation_rate_exp(bpp, num_bins, trials):
    """
    Function that performs an experiment on varying mutation rates, Mk. Like the 1st further experiment conducted,
    other parameters such as total number of trials and seeds used will be fixed and the same as other experiments.
    This is to ensure consistency in the amount of data produced for each experiment, and making reproducible results
    as well.
    :param bpp: The BPP problem name, in String format, to add to the plotting title.
    :param num_bins: Number of bins.
    :param trials: Number of trials.
    """
    # Create a list to store fitness scores from each mutation operator value, Mk, implemented on the EA.
    mutation_results = []

    # Create a list to data in a table format.
    cell_text = []

    # Mutation rates used will be 1, 2, 3, 4 and 5.# To ensure that results are consistent, the only changing variable
    # will be the mutation rate, Mk, nothing more.
    mutation_rates = [1, 2, 3, 4, 5]

    # Random seed generation will use values 1, 2, 3, 4 and 5.
    random_state = [1, 2, 3, 4, 5]

    # Create a list to store the 5 best fitness scores in correspondence to the 5 best solutions for 5 trials.
    best_fitness_scores = []

    # For each mutation rate value, run 5 trials using that value on the EA.
    for rate in mutation_rates:
        for i in range(trials):
            seed(random_state[i])

            print("Trial #" + str(i + 1) + ":")
            print("Seed Value is {}".format(random_state[i]))

            # Save the best solution, its fitness, and plotting data attained from a trial run and
            solution, sol_fitness, plot_data = ea_bpp(population_size=10,
                                                      tournament_size=2,
                                                      num_bins=num_bins,
                                                      num_items=500,
                                                      k=rate,
                                                      cross_over=True,
                                                      mutation=True)
            print("Final Fitness Score: " + str(sol_fitness))
            print()

            best_fitness_scores.append(sol_fitness)

        average_fitness = sum(best_fitness_scores) / len(best_fitness_scores)
        print("Average Fitness when using Mutation Rate, Mk, of Value k={}: {}".format(rate, average_fitness))
        print()
        print()
        mutation_results.append(round(average_fitness, 2))

    row_labels = [str(bpp) + " Mutation Rate (Mk) k = " + str(rate) for rate in mutation_rates]

    # Save results into a dataframe.
    mutation_results_df = pd.DataFrame(mutation_results, columns=["Average Fitness Scores"])

    # Iterate through dataframe and append all values into the list
    for each_row in range(len(mutation_results_df)):
        cell_text.append(mutation_results_df.iloc[each_row])

    # Plot results on a table.
    plt.table(cellText=cell_text,
              rowLabels=row_labels,
              colLabels=["Average Fitness Scores"],
              colWidths=[0.3],
              loc='center')
    plt.axis('tight')
    plt.axis('off')
    plt.show()


if __name__ == "__main__":

    # NOTE: If the lines of codes below is, in any case, ran in the Command Prompt or Terminal window, for every
    # experiment ran after 5 trials, a plotted figure will appear. The EA run for the next experiment will be
    # paused until the window (used to show the plotted figure) is either closed or saved.

    # Number of Bins for BPP1 (b = 10)
    bpp1_bin_num = 10

    # Number of Bins for BPP1 (b = 100)
    bpp2_bin_num = 100

    # Number of Bin Items = 500
    num_bin_items = 500

    print("RUNNING EVOLUTIONARY ALGORITHM ON BPP1 (Weight of Each Item i is 2i")
    run_all_bpp_experiments("BPP1", bpp1_bin_num)
    print()
    print()

    print("RUNNING EVOLUTIONARY ALGORITHM ON BPP2 (Weight of Each Item i is 2(i**2)")
    run_all_bpp_experiments("BPP2", bpp2_bin_num)
    print()
    print()

    print("Running Further Experiment on Varying Tournament Sizes (BPP1)")
    varying_tour_size_exp("BPP1", num_bins=10, trials=5)
    print()

    print("Running Further Experiment on Varying Tournament Sizes (BPP2)")
    varying_tour_size_exp("BPP2", num_bins=100, trials=5)
    print()

    print("Running Further Experiment on Varying Mutation Rate, Mk (BPP1)")
    varying_mutation_rate_exp("BPP1", num_bins=10, trials=5)
    print()

    print("Running Further Experiment on Varying Mutation Rate, Mk (BPP2)")
    varying_mutation_rate_exp("BPP2", num_bins=100, trials=5)
    print()
