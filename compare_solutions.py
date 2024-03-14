from typing import List, Dict, Any
from argparse import ArgumentParser
import pickle

import numpy as np


def print_stats(data, hist=False) -> float:
    data_mean: float = float(np.mean(data))
    print("Min/Max/Median/Mean(Std) %f/%f/%f/%f(%f)" % (min(data), max(data), float(np.median(data)),
                                                        data_mean, float(np.std(data))))
    if hist:
        hist1 = np.histogram(data)
        for x, y in zip(hist1[0], hist1[1]):
            print("%s %s" % (x, y))

    return data_mean


def get_solved_vals(results: Dict[str, Any], key: str) -> List:
    if "solved" not in results.keys():
        return results[key]

    vals: List = [x for x, solved in zip(results[key], results["solved"]) if solved]
    return vals


def print_results(results):
    times = np.array(get_solved_vals(results, "times"))
    lens = np.array([len(x) for x in get_solved_vals(results, "actions")])
    num_nodes_generated = get_solved_vals(results, "num_nodes_generated")

    mean_vals: List[float] = []
    print("-Lengths-")
    mean_vals.append(print_stats(lens))
    if "solved" in results.keys():
        print("-%% Solved-")
        print(100.0 * np.mean(results["solved"]))
        mean_vals.append(np.mean(results["solved"]))
    mean_vals.append(0.0)
    print("-Nodes Generated-")
    mean_vals.append(print_stats(num_nodes_generated))
    print("-Times-")
    mean_vals.append(print_stats(times))
    print("-Nodes/Sec-")
    mean_vals.append(print_stats(np.array(num_nodes_generated) / np.array(times)))

    print(",".join([str(x) for x in mean_vals]))

    if "iterations" in results.keys():
        print("-Itrs/Sec-")
        iterations = get_solved_vals(results, "iterations")
        print_stats(np.array(iterations) / np.array(times))


def main():
    # parse arguments
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--soln1', type=str, required=True, help="")
    parser.add_argument('--soln2', type=str, required=True, help="")

    args = parser.parse_args()

    results1 = pickle.load(open(args.soln1, "rb"))
    results2 = pickle.load(open(args.soln2, "rb"))

    lens1 = np.array([len(x) if x is not None else np.inf for x in results1["actions"]])
    lens2 = np.array([len(x) if x is not None else np.inf for x in results2["actions"]])

    print("%i states" % (len(results1["states"])))

    print("\n--SOLUTION 1---")
    print_results(results1)

    print("\n--SOLUTION 2---")
    print_results(results2)

    print("\n\n------Solution 2 - Solution 1 Lengths-----")
    print_stats(lens2 - lens1, hist=False)
    print("%.2f%% soln2 equal to soln1" % (100 * np.mean(lens2 == lens1)))


if __name__ == "__main__":
    main()
