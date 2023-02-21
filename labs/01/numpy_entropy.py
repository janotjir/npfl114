#!/usr/bin/env python3
import argparse
from typing import Tuple

import numpy as np

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--data_path", default="numpy_entropy_data.txt", type=str, help="Data distribution path.")
parser.add_argument("--model_path", default="numpy_entropy_model.txt", type=str, help="Model distribution path.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> Tuple[float, float, float]:
    # TODO: Load data distribution, each line containing a datapoint -- a string.
    data_dist = {}
    with open(args.data_path, "r") as data:
        for line in data:
            line = line.rstrip("\n")
            # TODO: Process the line, aggregating data with built-in Python
            # data structures (not NumPy, which is not suitable for incremental
            # addition and string mapping).
            if line not in data_dist: 
                data_dist[line] = 1
            else:
                data_dist[line] += 1

    # TODO: Load model distribution, each line `string \t probability`.
    model_dist = dict.fromkeys(data_dist.keys(), np.inf)
    with open(args.model_path, "r") as model:
        for line in model:
            line = line.rstrip("\n").split('\t')
            # TODO: process the line, aggregating using Python data structures
            if line[0] in model_dist.keys():
                model_dist[line[0]] = float(line[1])

    # TODO: Create a NumPy array containing the data distribution. The
    # NumPy array should contain only data, not any mapping. Alternatively,
    # the NumPy array might be created after loading the model distribution.
    # TODO: Create a NumPy array containing the model distribution.
    model_distribution = np.array(list(model_dist.values())).astype(np.float32)
    data_distribution = np.array(list(data_dist.values())).astype(np.float32) 
    data_distribution /= np.sum(data_distribution)

    # TODO: Compute the entropy H(data distribution). You should not use
    # manual for/while cycles, but instead use the fact that most NumPy methods
    # operate on all elements (for example `*` is vector element-wise multiplication).
    entropy = - np.sum(data_distribution*np.log(data_distribution))

    # TODO: Compute cross-entropy H(data distribution, model distribution).
    # When some data distribution elements are missing in the model distribution,
    # return `np.inf`.
    # TODO: Compute KL-divergence D_KL(data distribution, model_distribution),
    # again using `np.inf` when needed.
    crossentropy = np.inf
    kl_divergence = np.inf
    if np.isfinite(model_distribution).all():
        crossentropy = - np.sum(data_distribution*np.log(model_distribution))
        kl_divergence = np.sum(data_distribution*(np.log(data_distribution) - np.log(model_distribution)))

    # Return the computed values for ReCodEx to validate.
    return entropy, crossentropy, kl_divergence


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    entropy, crossentropy, kl_divergence = main(args)
    print("Entropy: {:.2f} nats".format(entropy))
    print("Crossentropy: {:.2f} nats".format(crossentropy))
    print("KL divergence: {:.2f} nats".format(kl_divergence))
