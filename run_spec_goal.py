from typing import List, cast, Dict, Any, Optional

from deepxube.environments.environment_abstract import EnvGrndAtoms, State
from deepxube.utils import env_select, program_utils, nnet_utils, viz_utils, data_utils
from deepxube.utils.timing_utils import Times
from search_state.spec_goal_asp import path_to_spec_goal
from deepxube.logic.program import Clause
from argparse import ArgumentParser
import torch
import os
import sys
import pickle
import numpy as np
import time


def main():
    # parse arguments
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--states', type=str, required=True, help="File containing states to solve")

    parser.add_argument('--env', type=str, required=True, help="")
    parser.add_argument('--bk_add', type=str, default="", help="File of additional background knowledge")

    parser.add_argument('--model_batch_size', type=int, default=1, help="Maximum number of models sampled at once "
                                                                        "for parallelized search")

    parser.add_argument('--results_dir', type=str, required=True, help="Directory to save results. Saves results after "
                                                                       "every instance.")

    parser.add_argument('--heur', type=str, required=True, help="nnet model file")
    parser.add_argument('--batch_size', type=int, default=100, help="Batch size for batch-weighted A* search")
    parser.add_argument('--weight', type=float, default=0.2, help="Weight on path cost f(n) = w * g(n) + h(n)")
    parser.add_argument('--max_search_itrs', type=float, default=100, help="Maximum number of iterations to search "
                                                                           "for a path to a given model.")

    parser.add_argument('--nnet_batch_size', type=int, default=None, help="Set to control how many states per GPU are "
                                                                          "evaluated by the neural network at a time. "
                                                                          "Does not affect path found, but will "
                                                                          "help if nnet is running out of memory.")

    parser.add_argument('--spec', type=str, required=True, help="Should have 'goal' in the head. "
                                                                "Separate multiple clauses by ';'")
    parser.add_argument('--spec_verbose', action='store_true', default=False, help="Set for verbose specification")
    parser.add_argument('--search_verbose', action='store_true', default=False, help="Set for verbose search")
    parser.add_argument('--viz_start', action='store_true', default=False, help="Set to visualize starting state")
    parser.add_argument('--viz_model', action='store_true', default=False, help="Set to visualize each model before "
                                                                                "search")
    parser.add_argument('--viz_goal', action='store_true', default=False, help="Set to visualize reached goal state")
    parser.add_argument('--redo', action='store_true', default=False, help="Set to start from scratch")
    parser.add_argument('--debug', action='store_true', default=False, help="Set when debugging with breakpoints")

    args = parser.parse_args()

    # Directory
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    # environment
    env: EnvGrndAtoms = cast(EnvGrndAtoms, env_select.get_environment(args.env))

    # get data
    data: Dict = pickle.load(open(args.states, "rb"))
    states: List[State] = data['states']

    results_file: str = "%s/results.pkl" % args.results_dir
    output_file: str = "%s/output.txt" % args.results_dir

    has_results: bool = False
    if os.path.isfile(results_file):
        has_results = True

    if has_results and (not args.redo):
        results: Dict[str, Any] = pickle.load(open(results_file, "rb"))
        if not args.debug:
            sys.stdout = data_utils.Logger(output_file, "a")
    else:
        results: Dict[str, Any] = {"states": states, "actions": [], "path_costs": [],
                                   "ASP init times": [], "model times": [], "search times": [], "check times": [],
                                   "superset times": [], "times": [], "num models init": [], "num models superset": [],
                                   "solved": []}

        if not args.debug:
            sys.stdout = data_utils.Logger(output_file, "w")

    # spec clauses
    spec_clauses_str = args.spec.split(";")
    clauses: List[Clause] = []
    for clause_str in spec_clauses_str:
        clause = program_utils.parse_clause(clause_str)[0]
        clauses.append(clause)
    print("Parsed input clauses:")
    print(clauses)

    # heuristic function
    device, devices, on_gpu = nnet_utils.get_device()
    print("device: %s, devices: %s, on_gpu: %s" % (device, devices, on_gpu))
    heuristic_fn = nnet_utils.load_heuristic_fn(args.heur, device, on_gpu, env.get_v_nnet(),
                                                env, clip_zero=True, batch_size=args.nnet_batch_size)

    start_idx = len(results["actions"])
    for state_idx in range(start_idx, len(states)):
        # start state
        state = states[state_idx]
        if args.viz_start:
            print("Starting state visualization:")
            viz_utils.visualize_examples(env, [state])

        # find path to goal
        solved: bool = False
        path_cost: float = np.inf
        path_actions: Optional[List[int]] = None
        num_models_init: int = 0
        num_models_superset: int = 0
        times: Times = Times()
        start_time = time.time()
        while not solved:
            (solved, state_path, path_actions, path_cost, num_mod_init_i, num_mod_sup_i,
             times_i) = path_to_spec_goal(env, state, clauses, heuristic_fn, args.model_batch_size, args.batch_size,
                                          args.weight, args.max_search_itrs, bk_add=args.bk_add,
                                          spec_verbose=args.spec_verbose, search_verbose=args.search_verbose,
                                          viz_model=args.viz_model)
            num_models_init += num_mod_init_i
            num_models_superset += num_mod_sup_i
            times.add_times(times_i)
            if solved and args.viz_goal:
                viz_utils.visualize_examples(env, [state_path[-1]])
        tot_time = time.time() - start_time

        results["actions"].append(path_actions)
        results["path_costs"].append(path_cost)
        results["num models init"].append(num_models_init)
        results["num models superset"].append(num_models_superset)
        results["ASP init times"].append(times.times['ASP init'])
        results["model times"].append(times.times['Model samp'])
        results["search times"].append(times.times['Search'])
        results["check times"].append(times.times['Check'])
        results["superset times"].append(times.times['Model superset'])
        results["times"].append(tot_time)
        results["solved"].append(solved)
        print(times.get_time_str())
        print(f"State: %i, PathCost: %.2f, # Models init: {num_models_init}, # Models superset: {num_models_superset}, "
              f"Solved: {solved}, Time: %.2f" % (state_idx, path_cost, tot_time))
        print(f"Means, SolnCost: %.2f, # Models init: %.2f, # Models superset: %.2f, Solved: %.2f%%, "
              f"Times - ASP init: %.2f, Model: %.2f, Search: %.2f, Check: %.2f, Superset: %.2f, "
              f"Tot: %.2f" % (_get_mean(results, "path_costs"), _get_mean(results, "num models init"),
                              _get_mean(results, "num models superset"), 100.0 * np.mean(results["solved"]),
                              _get_mean(results, "ASP init times"), _get_mean(results, "model times"),
                              _get_mean(results, "search times"), _get_mean(results, "check times"),
                              _get_mean(results, "superset times"), _get_mean(results, "times")))
        print("")


def _get_mean(results: Dict[str, Any], key: str) -> float:
    vals: List = [x for x, solved in zip(results[key], results["solved"]) if solved]
    if len(vals) == 0:
        return 0
    else:
        mean_val = np.mean([x for x, solved in zip(results[key], results["solved"]) if solved])
        return float(mean_val)


if __name__ == "__main__":
    main()
