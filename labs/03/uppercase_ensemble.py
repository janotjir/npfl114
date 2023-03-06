#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from uppercase_data import UppercaseData

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--test", action='store_true', default=False)
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")

# Team members:
# 4c2c10df-00be-4008-8e01-1526b9225726
# dc535248-fa6c-4987-b49f-25b6ede7c87d


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.debug:
        tf.config.run_functions_eagerly(True)
        tf.data.experimental.enable_debug_mode()

    '''eval acc: 99.079'''
    model_paths = ["logs/uppercase_v2.py-2023-03-02_171134-as=256,bs=32,d=False,e=5,m=,mp=2layer_10_model_ep5.h5,s=42,t=1,w=10/2layer_10_model_ep5.h5",
    "logs/uppercase_v2.py-2023-03-02_172324-as=128,bs=256,d=False,e=5,m=2layer_10,mp=None,s=42,t=8,w=10/2layer_10_ep5.h5",
    "logs/uppercase_v2.py-2023-03-02_194832-as=100,bs=256,d=False,e=10,m=2layer_10,mp=None,s=42,t=8,w=10/2layer_10_ep7.h5",
    "logs/uppercase_v2.py-2023-03-02_220607-as=100,bs=512,d=False,e=8,m=2layer_10_drop,mp=None,s=42,t=8,w=10/2layer_10_drop_ep8.h5",
    "logs/uppercase_v2.py-2023-03-02_220607-as=100,bs=512,d=False,e=8,m=2layer_10_drop,mp=None,s=42,t=8,w=10/2layer_10_drop_ep6.h5",
    "logs/epoch5.h5"]
    alphahabet_sizes = [256, 128, 100, 100, 100, 100]
    window_sizes = [10, 10, 10, 10, 10, 10]

    uppercase_data = UppercaseData(window_sizes[0], alphahabet_sizes[0])
    if args.test:
        output = tf.zeros([uppercase_data.test.data["windows"].shape[0], 2])
    else:
        output = tf.zeros([uppercase_data.dev.data["windows"].shape[0], 2])

    for i in tqdm(range(len(model_paths))):
        uppercase_data = UppercaseData(window_sizes[i], alphahabet_sizes[i])
        model = tf.keras.models.load_model(model_paths[i])

        if args.test:
            output += model.predict(uppercase_data.test.data["windows"], verbose=0)
        else:
            output += model.predict(uppercase_data.dev.data["windows"], verbose=0)
    output /= len(model_paths)

    if not args.test:
        ensemble_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        ensemble_accuracy.update_state(uppercase_data.dev.data["labels"], output)
        ensemble_accuracy = ensemble_accuracy.result().numpy()
        print(f"Ensemble accuracy on dev data: {ensemble_accuracy}")
    else:
        def selective_upper(index, char, labels):
            if labels[index] > 0 or index == 0:
                char = char.upper()
            return char

        os.makedirs("ensemble_results", exist_ok=True)
        with open(os.path.join("ensemble_results", "uppercase_test.txt"), "w", encoding="utf-8") as predictions_file:
            ch_list = [*uppercase_data.test.text]
            output = tf.math.argmax(output, axis=1)
            u_list = [selective_upper(index, char, output) for index, char in tqdm(enumerate(ch_list))]
            u_text = "".join(u_list)
            predictions_file.write(u_text)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)