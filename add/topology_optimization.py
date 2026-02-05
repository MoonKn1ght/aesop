"""
Random search for topology for benchmarking
"""
import time
import autograd.numpy as np
import matplotlib.pyplot as plt
import copy
import ray
import random
import pathlib
import scipy.optimize
from autograd.numpy.numpy_boxes import ArrayBox

import networkx as nx

from lib.minimal_save import extract_minimal_graph_info, build_from_minimal_graph_info
from algorithms.functions import logbook_update, logbook_initialize
from .parameter_optimization import parameters_optimize
from algorithms.speciation import Speciation, NoSpeciation, SimpleSubpopulationSchemeDist, vectorDIFF, photoNEAT, EditDistance
from algorithms.assets.graph_edit_distance import similarity_full_ged, similarity_reduced_ged

SPECIATION_MANAGER = NoSpeciation() 

def topology_optimization(graph, propagator, evaluator, evolver, io,
                          crossover_maker=None, parameter_opt_method='L-BFGS+GA',
                          ga_opts=None, update_rule='random', elitism_ratio=0,
                          target_species_num=4, protection_half_life=None,
                          cluster_address=None, local_mode=False, include_dashboard=False,
                          save_all_minimal_graph_data=True, save_all_minimal_hof_data=True,
                          ged_threshold_value=1.5):
    io.init_logging()
    log, log_metrics = logbook_initialize()

    if update_rule == 'random':
        update_population = update_population_topology_random  # set which update rule to use
    elif update_rule == 'preferential':
        update_population = update_population_topology_preferential
    elif update_rule == 'roulette':
        update_population = update_population_topology_roulette
    elif update_rule == 'tournament':
        update_population = update_population_topology_tournament
    elif update_rule == 'roulette edit distance':
        update_population = update_population_topology_roulette_editDistance
    elif update_rule == 'tournament edit distance':
        update_population = update_population_topology_tournament_editDistance
    elif update_rule == 'random simple subpop scheme':
        update_population = update_population_topology_random_simple_subpopulation_scheme
    elif update_rule == 'preferential simple subpop scheme':
        update_population = update_population_topology_preferential_simple_subpopulation_scheme
    elif update_rule == 'random vectorDIFF':
        update_population = update_population_topology_random_vectorDIFF
    elif update_rule == 'preferential vectorDIFF':
        update_population = update_population_topology_preferential_vectorDIFF
    elif update_rule == 'random photoNEAT':
        update_population = update_population_topology_random_photoNEAT
    elif update_rule == 'preferential photoNEAT':
        update_population = update_population_topology_preferential_photoNEAT
    else:
        raise NotImplementedError("This topology optimization update rule is not implemented yet. current options are 'random'")

    # start up the multiprocessing/distributed processing with ray, and make objects available to nodes
    if local_mode: print(f"Running in local_mode - not running as distributed computation")
    # ray.init(address=cluster_address, num_cpus=ga_opts['num_cpus'], local_mode=local_mode, ignore_reinit_error=True) #, object_store_memory=1e9)
    ray.init(address=cluster_address, num_cpus=ga_opts['num_cpus'], local_mode=local_mode, include_dashboard=include_dashboard, ignore_reinit_error=True) #, object_store_memory=1e9)
    evaluator_id, propagator_id = ray.put(evaluator), ray.put(propagator)

    # save the objects for analysis later
    io.save_json(ga_opts, 'ga_opts.json')
    for (object_filename, object_to_save) in zip(('propagator', 'evaluator', 'evolver'), (propagator, evaluator, evolver)):
        io.save_object(object_to_save=object_to_save, filename=f"{object_filename}.pkl")
    
    #save optimization settings for analysis later
    optimization_settings = {'update_rule':update_rule, 'target_species_num':target_species_num, 'protection_half_life':protection_half_life,\
                             'evolver': evolver.__class__.__name__, 'evaluator': evaluator.__class__.__name__}
    io.save_json(optimization_settings, 'optimization_settings.json')

    # create initial population and hof
    hof = init_hof(ga_opts['n_hof'])
    population = []
    score, graph = parameters_optimize_complete((None, graph), evaluator, propagator, method=parameter_opt_method)  # core progress of parameter optimization
    graph.score = score
    plot_graph(graph,propagator, evaluator, io)

    for individual in range(ga_opts['n_population']):
        population.append((score, copy.deepcopy(graph)))
    # do we want to save a reduced file of each graph throughout optimization?
    if save_all_minimal_graph_data:
        io.join_to_save_path("reduced_graphs").mkdir(parents=True, exist_ok=True)
        save_minimal_graph_data_pop([(score, graph)], io, gen=0, subfolder='reduced_graphs', filename_prefix='graph_')

    # do we want to save a reduced file of each HoF graph throughout optimization?
    if save_all_minimal_hof_data:
        io.join_to_save_path("reduced_hof_graphs").mkdir(parents=True, exist_ok=True)
        save_minimal_graph_data_pop([(score, graph)], io, gen=0, subfolder='reduced_hof_graphs', filename_prefix='hof_')

    t1 = time.time()
    for generation in range(0, ga_opts['n_generations']):
        print(f'\ngeneration {generation} of {ga_opts["n_generations"]}: time elapsed {time.time()-t1}s')

        if generation != 0: # we want population update, and selection rules to depend on the speciated fitness (as to favour rarer individuals)
            SPECIATION_MANAGER.speciate(population)
            SPECIATION_MANAGER.execute_fitness_sharing(population, generation)

        population = update_population(population, evolver, evaluator, target_species_num=target_species_num, # ga_opts['n_population'] / 20,
                                                                       protection_half_life=protection_half_life,
                                                                       crossover_maker=crossover_maker,
                                                                       elitism_ratio=elitism_ratio,
                                                                       verbose=ga_opts['verbose'],
                                                                       generation=generation / (ga_opts['n_generations']-1))
        if generation != 0:
            SPECIATION_MANAGER.reverse_fitness_sharing(population, generation)

        if ga_opts['verbose']:
            print(f'population length after update: {len(population)}')

        # optimize parameters on each node/CPU
        population = ray.get([parameters_optimize_multiprocess.remote(ind,
                                                                      evaluator_id,
                                                                      propagator_id,
                                                                      method=parameter_opt_method,
                                                                      verbose=ga_opts['verbose'])
                              for ind in population])
        save_scores_to_graph(population) # necessary for some algorithms
        hof = update_hof(hof=hof, population=population, verbose=ga_opts['verbose'], threshold_value=ged_threshold_value) # update before speciation, since we don't want this hof score affected by speciation

        SPECIATION_MANAGER.speciate(population) # we want the elements for the next generation to be picked as to maintain genetic diversity
        SPECIATION_MANAGER.execute_fitness_sharing(population, generation)
        population.sort(reverse = False, key=lambda x: x[0])  # we sort ascending, and take first (this is the minimum, as we minimizing)
        population = population[0:ga_opts['n_population']] # get rid of extra params, if we have too many
        SPECIATION_MANAGER.reverse_fitness_sharing(population, generation)
        for (score, graph) in population:
            graph.clear_propagation()

        # update logbook and hall of fame
        logbook_update(generation, population, log, log_metrics, time=(time.time()-t1), best=hof[0][0], verbose=ga_opts['verbose'])

        if save_all_minimal_graph_data:
            save_minimal_graph_data_pop(population, io, gen=generation+1, subfolder='reduced_graphs', filename_prefix='graph_')
        if save_all_minimal_hof_data:
            save_minimal_graph_data_pop(hof, io, gen=generation+1, subfolder='reduced_hof_graphs', filename_prefix='hof_')

    io.save_object(log, 'log.pkl')

    io.close_logging()
    evolver.close()
    return hof, log # hof is actually a list in and of itself, so we only look at the top element


def save_minimal_graph_data_pop(population, io, gen, subfolder='', filename_prefix=''):
    for ind, (score, graph) in enumerate(population):
        if score is None or graph is None:
            continue
        filename = pathlib.Path(subfolder).joinpath(filename_prefix+"gen{}_ind{}.json".format(gen, ind))
        json_data = extract_minimal_graph_info(graph)
        json_data['current_uuid'] = graph.current_uuid.hex
        json_data['parent_uuid'] = graph.parent_uuid.hex
        json_data['latest_mutation'] = graph.latest_mutation
        json_data['score'] = score
        io.save_json(json_data, filename)


def save_scores_to_graph(population):
    for (score, graph) in population:
        graph.score = score


def init_hof(n_hof):
    hof = [(None, None) for i in range(n_hof)]
    return hof


def update_hof(hof, population, similarity_measure='reduced_ged', threshold_value=1.5, verbose=False):
    """
    :param hof: list of N tuples, where each tuple is (score, graph) and are the best performing of the entire run so far
    :param population: current population of graphs, list of M tuples, with each tuple (score, graph)
    :param similarity_measure: string identifier for which similarity/distance measure to use. currently implemented are
        'reduced_ged': a graph reduction method before using the Graph Edit Distance measurement,
        'full_ged': directly using the Graph Edit Distance measurement on the system graphs
    :param threshold_value: positive float. two graphs with similarity below this value are considered to be the same structure
    :param verbose: debugging to print, boolean
    :return: returns the updated hof, list of tuples same as input
    """
    if similarity_measure == 'reduced_ged':
        similarity_function = similarity_reduced_ged
    elif similarity_measure == 'full_ged':
        similarity_function = similarity_full_ged
    else:
        raise NotImplementedError('this is not an implemented graph measure function. please use reduced_ged or full_ged.')


    for i, (score, graph) in enumerate(population):
        if verbose: print(f'\n\nNow checking with population index {i}')

        insert = False
        insert_ind = None
        remove_ind = -1
        check_similarity = True
        for j, (hof_j_score, hof_j) in enumerate(hof):
            if verbose: print(f'checking score against index {j} of the hof')
            if hof_j_score is None:
                insert = True
                insert_ind = j
                check_similarity = True
                break

            # if performing better, go to the next hof candidate with the next best score
            elif score < hof_j_score:
                insert = True
                insert_ind = j
                check_similarity = True
                if verbose: print(f'Better score than index {j} of the hof')
                break

            else:
                # no need to check similarity if the score is worse than all hof graphs
                check_similarity = False

        if not check_similarity:
            if verbose: print(f'There is no need to check the similarity')

        if check_similarity and (similarity_measure is not None):
            # similarity check with all HoF graphs
            for k, (hof_k_score, hof_k) in enumerate(hof):
                if verbose: print(f'Comparing similarity of with population index {i} with hof index {k}')

                if hof_k is not None:
                    sim = similarity_function(graph, hof_k)

                    if sim <= threshold_value or np.isclose(sim, threshold_value):
                        # there is another, highly similar graph in the hof
                        if k < j:
                            # there is a similar graph with a better score, do not add graph to hof
                            insert = False
                            if verbose: print(f'A similar, but better performing graph exists - the current graph will not be added')
                            break # breaks out of 'k' loop
                        elif k >= j:
                            # there is a similar graph with a worse score, add in that location instead
                            insert = True
                            remove_ind = k
                            if verbose: print(f'A similar graph exists at index {k} of the hof. The current graph scores better and will be added')
                            break
                    else:
                        if verbose: print(f'Similarity of {sim} is not below the threshold')
                        pass
        if insert: # places this graph into the insert_ind index in the halloffame
            hof[remove_ind] = 'x'
            hof.insert(insert_ind, (score, copy.deepcopy(graph)))
            hof.remove('x')
            if verbose: print(f'Replacing HOF individual {remove_ind}, new score of {score}')
        else:
            if verbose: print(f'Not adding population index {i} into the hof')

    return hof


def update_population_topology_random(population, evolver, evaluator, elitism_ratio=0, generation=None, **hyperparameters):
    # implement elitism
    population_pass_index = round(elitism_ratio * len(population))
    if population_pass_index > len(population):
        population_pass_index == len(population)
    
    pass_population = copy.deepcopy(population[0:population_pass_index])
    
    # mutating the population occurs on head node, then graphs are distributed to nodes for parameter optimization
    for i, (score, graph) in enumerate(population):
        while True:
            graph_tmp, evo_op_choice = evolver.evolve_graph(graph, evaluator, generation=generation)
            x0, node_edge_index, parameter_index, *_ = graph_tmp.extract_parameters_to_list()
            try:
                graph_tmp.assert_number_of_edges()
            except:
                print(f'Could not evolve this graph.{[node for node in graph_tmp.nodes()]}\n {graph}')
                continue

            if len(x0) == 0:
                continue
            else:
                graph = graph_tmp
                break
        population[i] = (None, graph)

    return population + pass_population


def update_population_topology_preferential(population, evolver, evaluator, preferentiality_param=2, elitism_ratio=0.1, generation=None, **hyperparameters):
    """
    Updates population such that the fitter individuals have a larger chance of reproducing

    So we want individuals to reproduce about once on average, more for the fitter, less for the less fit
    Does not modify the original graphs in population, unlike the random update rule

    :pre-condition: population is sorted in ascending order of score (i.e. most fit to least fit)
    """
    most_fit_reproduction_mean = 2 # most fit element will on average reproduce this many additional times (everyone reproduces once at least)
    
    # 1. Initialize scores (only happens on generation 0)
    if (population[0][0] is not None):
        score_array = np.array([score for (score, _) in population])
    else:
        score_array = np.ones(len(population)).reshape(len(population), 1)
        most_fit_reproduction_mean = 1
    
    new_pop = []

    # 2. Mutate existing elements (fitter individuals have a higher expectation value for number of reproductions)
    # basically after the initial reproduction, the most fit reproduces on average <most_fit_reproduction_mean> times, and the least fit never reproduces
    
    # TODO: test different ways of getting this probability (one that favours the top individuals less?)
    break_probability = 1 / most_fit_reproduction_mean * (score_array / np.min(score_array))**(1 / preferentiality_param)

    print(f'break prob: {break_probability}')

    for i, (score, graph) in enumerate(population):
        x, node_edge_index, parameter_index, *_ = graph.extract_parameters_to_list()
        parent_parameters = copy.deepcopy(x)
        while True:
            graph_tmp, evo_op_choice = evolver.evolve_graph(copy.deepcopy(graph), evaluator, generation=generation)
            try:
                graph.distribute_parameters_from_list(parent_parameters, node_edge_index, parameter_index) # This is a hacky fix because smh graph parameters are occasionally modified through the deepcopy
            except IndexError as e:
                print(e)
                print(graph)
                print(f'parent parameters: {parent_parameters}')
                raise e

            x0, node_edge_index, parameter_index, *_ = graph_tmp.extract_parameters_to_list()
            try:
                graph_tmp.assert_number_of_edges()
            except:
                continue

            if len(x0) == 0:
                continue
            
            new_pop.append((None, graph_tmp))
            if random.random() < break_probability[i]:
                break
    
    population_pass_index = round(elitism_ratio * len(population))
    if population_pass_index > len(population):
        population_pass_index == len(population)

    return new_pop + population[0:population_pass_index]


def update_population_topology_roulette(population, evolver, evaluator, preferentiality_param=1, elitism_ratio=0.1, generation=None, **hyperparameters):
    """
    Updates population such that the fitter individuals have a larger chance of reproducing, using the "roulette wheel" method

    So we want individuals to reproduce about once on average, more for the fitter, less for the less fit
    Does not modify the original graphs in population, unlike the random update rule

    :pre-condition: population is sorted in ascending order of score (i.e. most fit to least fit)
    :param preferentiality_param: the larger the parameter, the more fitter individuals (lower score) are favoured
    """    
    # 1. Initialize scores (only happens on generation 0)
    if (population[0][0] is not None):
        score_array = np.array([score for (score, _) in population])
    else:
        score_array = np.ones(len(population)).reshape(len(population), 1)
    
    new_pop = []

    # 2. Build roulette
    probability_roulette = 1 / score_array**preferentiality_param
    probability_roulette /= np.sum(probability_roulette)
    print(f'scores: {score_array}')
    print(f'probablities: {probability_roulette}')

    # 3. Sample using roulette
    for _ in range(len(population)):
        reproduction_index = np.random.choice(np.arange(0, len(population)), p=probability_roulette)
        graph = population[reproduction_index][1]
        graph_tmp, _ = evolver.evolve_graph(copy.deepcopy(graph), evaluator, generation=generation)
        graph_tmp.assert_number_of_edges()
        # TODO: determine whether the following is a correct check to make (or is it just alright?)
        # x0, node_edge_index, parameter_index, *_ = graph_tmp.extract_parameters_to_list()
        # if (len(x0)) == 0:
        #     raise ValueError('This graph has no parameters')
        new_pop.append((None, graph_tmp))
    
    population_pass_index = round(elitism_ratio * len(population))
    if population_pass_index > len(population):
        population_pass_index == len(population)

    return new_pop + population[0:population_pass_index]


def update_population_topology_tournament(population, evolver, evaluator, tournament_size_divisor=3, elitism_ratio=0.1, generation=None, **hyperparameters):
    """
    Updates population such that the fitter individuals have a larger chance of reproducing, using the "roulette wheel" method

    So we want individuals to reproduce about once on average, more for the fitter, less for the less fit
    Does not modify the original graphs in population, unlike the random update rule

    :pre-condition: population is sorted in ascending order of score (i.e. most fit to least fit)
    :param tournament_size_divisor: tournament size as ratio of total population size
                                    (e.g. for population size = 9, tournament_size_divisor=3, tournament size = 9 / 3 = 3)
    """    
    # 1. Initialize scores (only happens on generation 0)
    if (population[0][0] is not None):
        score_array = np.array([score for (score, _) in population])
    else:
        score_array = np.ones(len(population)).reshape(len(population), 1)
    
    new_pop = []

    # 2. Setup tournament
    tournament_size = len(population) // tournament_size_divisor
    if tournament_size == 0:
        print(f'WARNING: tournament size is zero. Setting size to 1')
        tournament_size = 1

    # 3. Sample using tournament
    for _ in range(len(population)):
        tournament_indices = np.random.choice(np.arange(0, len(population)), size=tournament_size)
        graph = population[min(tournament_indices)][1] # this works, because the vals were already sorted by fitness
        graph_tmp, _ = evolver.evolve_graph(copy.deepcopy(graph), evaluator, generation=generation)
        graph_tmp.assert_number_of_edges()
        # TODO: determine whether the following is a correct check to make (or is it just alright?)
        # x0, node_edge_index, parameter_index, *_ = graph_tmp.extract_parameters_to_list()
        # if (len(x0)) == 0:
        #     raise ValueError('This graph has no parameters')
        new_pop.append((None, graph_tmp))
    
    population_pass_index = round(elitism_ratio * len(population))
    if population_pass_index > len(population):
        population_pass_index == len(population)

    return new_pop + population[0:population_pass_index]


# ------------------------------- Speciation setup helpers -----------------------------------

def _simple_subpopulation_setup(population, **hyperparameters):
    global SPECIATION_MANAGER
    # set up speciation if it's not set up already. This is only necessary on first function call
    if population[0][1].speciation_descriptor is None:
        dist_func = SimpleSubpopulationSchemeDist().distance
        SPECIATION_MANAGER = Speciation(target_species_num=hyperparameters['target_species_num'],
                                        protection_half_life=hyperparameters['protection_half_life'],
                                        distance_func=dist_func, 
                                        verbose=hyperparameters['verbose'])
        for i, (_, graph) in enumerate(population):
            graph.speciation_descriptor = {'name':'simple subpopulation scheme', 'label':i % hyperparameters['target_species_num']}


def _vectorDIFF_setup(population, **hyperparameters):
    global SPECIATION_MANAGER
    # set up speciation if it's not set up already, noly necessary on first function call
    if population[0][1].speciation_descriptor is None:
        dist_func = vectorDIFF().distance
        SPECIATION_MANAGER = Speciation(target_species_num=hyperparameters['target_species_num'],
                                        protection_half_life=hyperparameters['protection_half_life'],
                                        distance_func=dist_func, 
                                        verbose=hyperparameters['verbose'])
        for _, graph in population:
            graph.speciation_descriptor = {'name':'vectorDIFF'}


def _photoNEAT_setup(population, **hyperparameters):
    global SPECIATION_MANAGER
    # set up speciation if it's not set up already, only necessary on first function call
    if population[0][1].speciation_descriptor is None:
        dist_func = photoNEAT().distance
        SPECIATION_MANAGER = Speciation(target_species_num=hyperparameters['target_species_num'],
                                        protection_half_life=hyperparameters['protection_half_life'],
                                        distance_func=dist_func, 
                                        verbose=hyperparameters['verbose'])
        for i, (_, graph) in enumerate(population):
            marker_node_map = {}
            node_marker_map = {}
            for i, node in enumerate(graph.nodes): # we only expect 2 things to start with, so i = 0, 1
                marker_node_map[i] = node
                node_marker_map[node] = i
            graph.speciation_descriptor = {'name':'photoNEAT', 'marker to node':marker_node_map, 'node to marker':node_marker_map}

def _edit_distance_setup(population, **hyperparameters):
    global SPECIATION_MANAGER
    if population[0][1].speciation_descriptor is None:
        dist_func = EditDistance().distance
        SPECIATION_MANAGER = Speciation(target_species_num=None,
                                        protection_half_life=hyperparameters['protection_half_life'],
                                        distance_func=dist_func,
                                        verbose=hyperparameters['verbose'])
        for i, (_, graph) in enumerate(population):
            graph.speciation_descriptor = {'name':'editDistance'}

# ---------------------------------- Speciated population update ------------------------------
def update_population_topology_random_simple_subpopulation_scheme(population, evolver, evaluator, **hyperparameters):
    _simple_subpopulation_setup(population, **hyperparameters)
    return update_population_topology_random(population, evolver, evaluator, **hyperparameters) # rest goes on normally


def update_population_topology_preferential_simple_subpopulation_scheme(population, evolver, evaluator, **hyperparameters):
    _simple_subpopulation_setup(population, **hyperparameters)
    return update_population_topology_preferential(population, evolver, evaluator, **hyperparameters) # rest goes on normally


def update_population_topology_random_vectorDIFF(population, evolver, evaluator, **hyperparameters):
    _vectorDIFF_setup(population, **hyperparameters)
    return update_population_topology_random(population, evolver, evaluator, **hyperparameters)


def update_population_topology_preferential_vectorDIFF(population, evolver, evaluator, **hyperparameters):
    _vectorDIFF_setup(population, **hyperparameters)
    return update_population_topology_preferential(population, evolver, evaluator, **hyperparameters)


def update_population_topology_random_photoNEAT(population, evolver, evaluator, **hyperparameters):
    """
    NOTE: with random population updates, NEAT speciation will not be very useful! That's because each individual just
    mutates once and moves on, which means that the speciation will brutally branch out and never recover

    Would be more useful with crossovers, or a number of offspring proportional which depends on their fitness
    """
    _photoNEAT_setup(population, **hyperparameters)
    return update_population_topology_random(population, evolver, evaluator, **hyperparameters)


def update_population_topology_preferential_photoNEAT(population, evolver, evaluator, **hyperparameters):
    _photoNEAT_setup(population, **hyperparameters)
    return update_population_topology_preferential(population, evolver, evaluator, **hyperparameters)


def update_population_topology_preferential_editDistance(population, evolver, evaluator, **hyperparameters):
    _edit_distance_setup(population, **hyperparameters)
    return update_population_topology_preferential(population, evolver, evaluator, **hyperparameters)


def update_population_topology_tournament_editDistance(population, evolver, evaluator, **hyperparameters):
    _edit_distance_setup(population, **hyperparameters)
    return update_population_topology_tournament(population, evolver, evaluator, **hyperparameters)


def update_population_topology_roulette_editDistance(population, evolver, evaluator, **hyperparameters):
    _edit_distance_setup(population, **hyperparameters)
    return update_population_topology_roulette(population, evolver, evaluator, **hyperparameters)


def parameters_optimize_complete(ind, evaluator, propagator, method='', verbose=True):
    score, graph = ind
    if score is not None:
        raise RuntimeError('The score should initially be None')

    try:
        graph.update_graph()  # updates propagation order and input/outputs on nodes
        graph.clear_propagation()
        # graph.sample_parameters(probability_dist='uniform', **{'triangle_width': 0.1})  # removed to keep old parameters
        x0, model, parameter_index, *_ = graph.extract_parameters_to_list()     # TODO: 6 PM appear here. x0 indicates the initial parameters.
        graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=True)
        if len(x0) == 0:
            return graph.func(x0), graph
        graph, parameters, score, log = parameters_optimize(graph, x0=x0, method=method, verbose=verbose)
        # TODO: RuntimeWarning: overflow encountered in true_divide
        graph.scaled_hess_matrix = graph.hess(parameters)  # we calculate this here as it takes a long time - shouldn't calculate again

        # if there are any ArrayBoxes (autgrad tracers) in the Graph object, it cannot save (recursion depth error)
        # so this is a quick hack, to just re-run to ensure that it is normal Python types in the model objects
        graph.func(parameters)  # removing this will cause errors in saving graph objects
        return score, graph

    except Exception as e:
        print(f'error caught in parameter optimization: {e}')
        raise e
        return 99999999, graph

@ray.remote
def parameters_optimize_multiprocess(ind, evaluator, propagator, method='NULL', verbose=True):
    return parameters_optimize_complete(ind, evaluator, propagator, method=method, verbose=verbose)


def save_hof(hof, io, type='full'):
    for i, (score, graph) in enumerate(hof):
        if type == 'full':
            io.save_object(object_to_save=graph.duplicate_and_simplify_graph(graph), filename=f"graph_hof{i}.pkl")
        elif type == 'minimal':
            raise Warning('Not implemented, reverting to full save')
            io.save_object(object_to_save=graph.duplicate_and_simplify_graph(graph), filename=f"graph_hof{i}.pkl")
    return


def plot_hof(hof, propagator, evaluator, io):
    # 获取所有sink节点名称并按数字排序
    sink_nodes = sorted([k for k in evaluator.targets.keys() if k.startswith('sink')],
                        key=lambda x: int(x[4:]))
    num_sinks = len(sink_nodes)

    # 动态计算子图列数 (拓扑图 + 各sink对比 + 分数注释)
    ncols = 1 + num_sinks + 1

    # 创建画布 (行数=hof数量，列数=动态计算)
    fig, axs = plt.subplots(nrows=len(hof), ncols=ncols,
                            figsize=(4 * ncols, 4 * len(hof)),
                            squeeze=False)  # 确保axs始终是二维数组

    # 扩展时域显示范围（显示完整时间轴）
    t_ns = propagator.t * 1e9  # 转换为纳秒单位
    xlim_full = [t_ns.min(), t_ns.max()]  # 完整时域范围

    for i, (score, graph) in enumerate(hof):
        # 执行信号传播
        graph.score = score
        graph.propagate(propagator, save_transforms=False)

        # 绘制拓扑图
        graph.draw(ax=axs[i, 0], debug=False)
        axs[i, 0].set_title(f"Topology #{i + 1}")
        #TODO: show the graph
        # graph.show()

        # 绘制各sink波形对比
        for col, sink in enumerate(sink_nodes, start=1):
            # 获取目标信号和实际信号
            target = evaluator.targets[sink]
            measured = np.abs(graph.measure_propagator(sink))

            # 绘制时域对比
            axs[i, col].plot(t_ns, target, '--', label=f'Target {sink}')
            axs[i, col].plot(t_ns, measured, '-', label=f'Measured {sink}')
            axs[i, col].set(xlim=xlim_full,  # 显示完整时域
                            xlabel='Time (ns)',
                            ylabel='Amplitude',
                            title=f'{sink} Comparison')
            axs[i, col].legend(loc='upper right', fontsize=8)

            # 添加性能指标注释
            mse = np.mean((target - measured) ** 2)
            axs[i, col].annotate(f"MSE: {mse:.2e}",
                                 xy=(0.95, 0.95),
                                 xycoords='axes fraction',
                                 ha='right', va='top',
                                 fontsize=8,
                                 bbox=dict(facecolor='white', alpha=0.8))

        # 分数注释列
        note_col = 1 + num_sinks  # 最后一列
        axs[i, note_col].axis('off')
        axs[i, note_col].annotate(
            f"HoF #{i + 1}\nScore: {score:.2e}\n"
            f"Components: {len(graph.nodes)}\n"
            f"Connections: {len(graph.edges)}",
            xy=(0.5, 0.5), xycoords='axes fraction',
            ha='center', va='center',
            fontsize=10,
            bbox=dict(facecolor='lavender', alpha=0.5)
        )
    plt.show()
    plt.tight_layout(pad=3.0)
    io.save_fig(fig=fig, filename='dynamic_hof.png')
    plt.close(fig)


def plot_graph(graph, propagator, evaluator, io):
    if not hasattr(evaluator, 'targets'):
        sink = 'sink'
        target = evaluator.target
        measured = np.abs(graph.measure_propagator(sink))

        fig, axs = plt.subplots(nrows=1, ncols=2,
                                figsize=(4 * 2, 4),
                                squeeze=False)  # 确保axs始终是二维数组

        # 扩展时域显示范围（显示完整时间轴）
        t_ns = propagator.t * 1e9  # 转换为纳秒单位
        xlim_full = [t_ns.min(), t_ns.max()]  # 完整时域范围

        graph.propagate(propagator, save_transforms=False)

        # 绘制拓扑图
        graph.draw(ax=axs[0, 0], debug=False)
        axs[0, 0].set_title(f"Topology")

        # 绘制时域对比
        axs[0, 1].plot(t_ns, target, '--', label=f'Target {sink}')
        axs[0, 1].plot(t_ns, measured, '-', label=f'Measured {sink}')
        axs[0, 1].set(xlim=xlim_full,  # 显示完整时域
                        xlabel='Time (ns)',
                        ylabel='Amplitude',
                        title=f'{sink} Comparison')
        axs[0, 1].legend(loc='upper right', fontsize=8)

        # 添加性能指标注释
        mse = np.mean((target - measured) ** 2)
        axs[0, 1].annotate(f"MSE: {mse:.2e}",
                             xy=(0.95, 0.95),
                             xycoords='axes fraction',
                             ha='right', va='top',
                             fontsize=8,
                             bbox=dict(facecolor='white', alpha=0.8))

    else:
        sink_nodes = sorted([k for k in evaluator.targets.keys() if k.startswith('sink')],
                        key=lambda x: int(x[4:]))
        num_sinks = len(sink_nodes)
        ncols = 1 + num_sinks

        fig, axs = plt.subplots(nrows=1, ncols=ncols,
                                figsize=(4 * ncols, 4 ),
                                squeeze=False)  # 确保axs始终是二维数组

        # 扩展时域显示范围（显示完整时间轴）
        t_ns = propagator.t * 1e9  # 转换为纳秒单位
        xlim_full = [t_ns.min(), t_ns.max()]  # 完整时域范围

        graph.propagate(propagator, save_transforms=False)

        # 绘制拓扑图
        graph.draw(ax=axs[0, 0], debug=False)
        axs[0, 0].set_title(f"Topology")

        # 绘制各sink波形对比
        for col, sink in enumerate(sink_nodes, start=1):
            # 获取目标信号和实际信号
            target = evaluator.targets[sink]
            measured = np.abs(graph.measure_propagator(sink))

            # 绘制时域对比
            axs[0, col].plot(t_ns, target, '--', label=f'Target {sink}')
            axs[0, col].plot(t_ns, measured, '-', label=f'Measured {sink}')
            axs[0, col].set(xlim=xlim_full,  # 显示完整时域
                            xlabel='Time (ns)',
                            ylabel='Amplitude (a.u)',
                            title=f'{sink} Comparison')
            axs[0, col].legend(loc='upper right', fontsize=8)

            # 添加性能指标注释
            mse = np.mean((target - measured) ** 2)
            axs[0, col].annotate(f"MSE: {mse:.2e}",
                                 xy=(0.95, 0.95),
                                 xycoords='axes fraction',
                                 ha='right', va='top',
                                 fontsize=8,
                                 bbox=dict(facecolor='white', alpha=0.8))
    plt.show()
    plt.tight_layout(pad=3.0)
    io.save_fig(fig=fig, filename='current_graph.png')
    plt.close(fig)
