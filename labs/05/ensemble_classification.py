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
from tqdm import tqdm

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=12, type=int, help="Maximum number of threads to use.")
parser.add_argument("--test", default=False, action="store_true", help="Run model on test dataset")


# Team members:
# 4c2c10df-00be-4008-8e01-1526b9225726
# dc535248-fa6c-4987-b49f-25b6ede7c87d


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    tf.keras.utils.set_random_seed(args.seed)
    # tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.debug:
        tf.config.run_functions_eagerly(True)
        tf.data.experimental.enable_debug_mode()

    args.logdir = "ensemble_cls_results"
    os.makedirs(args.logdir, exist_ok=True)

    # Load data
    cags = CAGS()

    generator = tf.random.Generator.from_seed(args.seed)
    def train_augment_tf_image(image: tf.Tensor, label: tf.Tensor):
        if generator.uniform([]) >= 0.5:
            image = tf.image.flip_left_right(image)
        image = tf.image.resize_with_crop_or_pad(image, CIFAR10.H + 6, CIFAR10.W + 6)
        image = tf.image.resize(image, [generator.uniform([], CIFAR10.H, CIFAR10.H + 12 + 1, dtype=tf.int32),
                                        generator.uniform([], CIFAR10.W, CIFAR10.W + 12 + 1, dtype=tf.int32)])
        image = tf.image.crop_to_bounding_box(
            image, target_height=CIFAR10.H, target_width=CIFAR10.W,
            offset_height=generator.uniform([], maxval=tf.shape(image)[0] - CIFAR10.H + 1, dtype=tf.int32),
            offset_width=generator.uniform([], maxval=tf.shape(image)[1] - CIFAR10.W + 1, dtype=tf.int32),
        )
        return image, label


    # 95.098
    model_paths = [
        "logs/cags_classification.py-2023-03-20_121930-a=False,bs=32,d=False,e=5,ls=0.0,mp=,s=42,t=False,t=1/ep5.h5",
        "logs/cags_classification.py-2023-03-20_132115-a=False,bs=32,d=False,e=10,ls=0.1,mp=,s=42,t=False,t=1/ep7.h5",
        "logs/cags_classification.py-2023-03-26_202329-a=False,bs=64,d=False,e=5,ls=0.2,mp=,s=42,t=False,t=1/ep5.h5",
        #"logs/notebook-2023-03-26_192349-a=False,bs=32,d=False,e=10,ls=0.2,mp=,s=42,t=False,t=1/ep8.h5",
        #"logs/notebook-2023-03-26_193625-a=False,bs=32,d=False,e=10,ls=0.2,mp=logs/ep5.h5",
        #"logs/notebook-2023-03-26_195911-a=False,bs=64,d=False,e=10,ls=0.3,mp=logs/ep5.h5"
    ]

    # 95.098
    model_paths = [
        "logs/cags_classification.py-2023-03-20_121930-a=False,bs=32,d=False,e=5,ls=0.0,mp=,s=42,t=False,t=1/ep5.h5",
        "logs/cags_classification.py-2023-03-20_132115-a=False,bs=32,d=False,e=10,ls=0.1,mp=,s=42,t=False,t=1/ep7.h5",
        #"logs/cags_classification.py-2023-03-26_202329-a=False,bs=64,d=False,e=5,ls=0.2,mp=,s=42,t=False,t=1/ep5.h5",
        "logs/notebook-2023-03-26_192349-a=False,bs=32,d=False,e=10,ls=0.2,mp=,s=42,t=False,t=1/ep8.h5",
        #"logs/notebook-2023-03-26_193625-a=False,bs=32,d=False,e=10,ls=0.2,mp=logs/ep5.h5",
        #"logs/notebook-2023-03-26_195911-a=False,bs=64,d=False,e=10,ls=0.3,mp=logs/ep5.h5"
    ]

    # 95.098
    model_paths = [
        "logs/cags_classification.py-2023-03-20_121930-a=False,bs=32,d=False,e=5,ls=0.0,mp=,s=42,t=False,t=1/ep5.h5",
        "logs/cags_classification.py-2023-03-20_132115-a=False,bs=32,d=False,e=10,ls=0.1,mp=,s=42,t=False,t=1/ep7.h5",
        #"logs/cags_classification.py-2023-03-26_202329-a=False,bs=64,d=False,e=5,ls=0.2,mp=,s=42,t=False,t=1/ep5.h5",
        "logs/notebook-2023-03-26_192349-a=False,bs=32,d=False,e=10,ls=0.2,mp=,s=42,t=False,t=1/ep8.h5",
        "logs/notebook-2023-03-26_193625-a=False,bs=32,d=False,e=10,ls=0.2,mp=logs/ep5.h5",
        #"logs/notebook-2023-03-26_195911-a=False,bs=64,d=False,e=10,ls=0.3,mp=logs/ep5.h5"
    ]

    # 94.77
    '''model_paths = [
        "logs/cags_classification.py-2023-03-20_121930-a=False,bs=32,d=False,e=5,ls=0.0,mp=,s=42,t=False,t=1/ep5.h5",
        "logs/cags_classification.py-2023-03-20_132115-a=False,bs=32,d=False,e=10,ls=0.1,mp=,s=42,t=False,t=1/ep7.h5",
        #"logs/cags_classification.py-2023-03-26_202329-a=False,bs=64,d=False,e=5,ls=0.2,mp=,s=42,t=False,t=1/ep5.h5",
        "logs/notebook-2023-03-26_192349-a=False,bs=32,d=False,e=10,ls=0.2,mp=,s=42,t=False,t=1/ep8.h5",
        "logs/notebook-2023-03-26_193625-a=False,bs=32,d=False,e=10,ls=0.2,mp=logs/ep5.h5",
        "logs/notebook-2023-03-26_195911-a=False,bs=64,d=False,e=10,ls=0.3,mp=logs/ep5.h5"
    ]'''

    # 94.77
    '''model_paths = [
        "logs/cags_classification.py-2023-03-20_121930-a=False,bs=32,d=False,e=5,ls=0.0,mp=,s=42,t=False,t=1/ep5.h5",
        "logs/cags_classification.py-2023-03-20_132115-a=False,bs=32,d=False,e=10,ls=0.1,mp=,s=42,t=False,t=1/ep7.h5",
        "logs/cags_classification.py-2023-03-26_202329-a=False,bs=64,d=False,e=5,ls=0.2,mp=,s=42,t=False,t=1/ep5.h5",
        "logs/notebook-2023-03-26_192349-a=False,bs=32,d=False,e=10,ls=0.2,mp=,s=42,t=False,t=1/ep8.h5",
        "logs/notebook-2023-03-26_193625-a=False,bs=32,d=False,e=10,ls=0.2,mp=logs/ep5.h5",
        #"logs/notebook-2023-03-26_195911-a=False,bs=64,d=False,e=10,ls=0.3,mp=logs/ep5.h5"
    ]'''

    if not args.test:
        dev = cags.dev.map(lambda x: x["image"])
        dev = dev.batch(args.batch_size)
 
        output = tf.zeros([306, len(CAGS.LABELS)])

        for i in tqdm(range(len(model_paths))):
            model = tf.keras.models.load_model(model_paths[i])
            with tf.device("/GPU:0"):
                output += model.predict(dev, verbose=0)
        
        output /= len(model_paths)
        output = tf.argmax(output, axis=1)
        output = [int(x) for x in output]

        result = CAGS.evaluate_classification(cags.dev, output)
        print(f"Ensemble accuracy on dev data: {result}")

    else:

        test = cags.test.map(lambda x: x["image"])
        test = test.batch(args.batch_size)

        output = tf.zeros([612, len(CAGS.LABELS)])

        for i in tqdm(range(len(model_paths))):
            model = tf.keras.models.load_model(model_paths[i])
            with tf.device("/GPU:0"):
                output += model.predict(test, verbose=0)
        
        output /= len(model_paths)

        # Generate test set annotations, but in `args.logdir` to allow parallel execution.
        with open(os.path.join(args.logdir, "cags_classification.txt"), "w", encoding="utf-8") as predictions_file:
            # TODO: Predict the probabilities on the test set

            for probs in output:
                print(np.argmax(probs), file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)