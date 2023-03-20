#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from cags_dataset import CAGS

import matplotlib.pyplot as plt

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=12, type=int, help="Maximum number of threads to use.")
parser.add_argument("--results_path", default="/home/jirkova1/npfl114/labs/05/logs/cags_segmentation_full_save.py-2023-03-20_160147-a=False,bs=32,e=5,mp=,s=42,t=False,t=1/", type=str, help="Specify path to results folder")


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.debug:
        tf.config.run_functions_eagerly(True)
        tf.data.experimental.enable_debug_mode()

        
    # Load data
    cags = CAGS()
    masks = np.load(args.results_path + "test_masks.npy")

    os.makedirs(args.results_path + "outputs", exist_ok=True)

    for i, example in enumerate(cags.test.as_numpy_iterator()):
        #plt.title(f"{CAGS.LABELS[int(labels[i])]}")
        img = example["image"]
        mask = masks[i, :, :]
        masked = img * mask
        print(i)
        plt.imshow(masked)
        plt.savefig(args.results_path + f"outputs/{i}.png")


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
