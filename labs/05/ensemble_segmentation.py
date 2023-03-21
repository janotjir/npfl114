#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from cags_dataset import CAGS
from cags_segmentation import EfficientUNetV2B0, UResEfficientNetV2B0, UResEfficientNetV2B0L

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
parser.add_argument("--save_masks", default=False, action='store_true', help="Specify if the mask should be saved in .npy format")


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

    args.logdir = "ensemble_seg_results"
    os.makedirs(args.logdir, exist_ok=True)

    # Load data
    cags = CAGS()

    # Convert images from tf.uint8 to tf.float32 and scale them to [0, 1] in the process.
    def image_to_float(image: tf.Tensor, label: tf.Tensor):
        return tf.image.convert_image_dtype(image, tf.float32), label

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

    # IoU on dev: 89.89
    '''model_types = ['ureseffnet', 'ureseffnetL']
    out_channels = [1, 1]
    model_paths = ["logs/ueffnet_bin/ep11.h5",
    "logs/ueffnetL_bin/ep11.h5"]'''

    # IoU on dev: 89.90
    '''model_types = ['effunet', 'ureseffnet', 'ureseffnetL']
    out_channels = [2, 1, 1]
    model_paths = ["logs/effunet_cat/ep9.h5",
    "logs/ueffnet_bin/ep11.h5",
    "logs/ueffnetL_bin/ep11.h5"]'''

    # IoU on dev: 89.94
    model_types = ['effunet', 'effunet', 'ureseffnet', 'ureseffnetL']
    out_channels = [2, 1, 1, 1]
    model_paths = ["logs/effunet_cat/ep9.h5",
    "logs/effunet_bin/ep5.h5",
    "logs/ueffnet_bin/ep11.h5",
    "logs/ueffnetL_bin/ep11.h5"]


    if not args.test:
        data = cags.dev.map(lambda x: x["image"])
        data = data.batch(args.batch_size)

        output = tf.zeros((306, 224, 224))

        for i in tqdm(range(len(model_paths))):
            # load model
            if model_types[i] == 'effunet':
                model = EfficientUNetV2B0(out_channels=out_channels[i])
            elif model_types[i] == 'ureseffnet':
                model = UResEfficientNetV2B0(out_channels=out_channels[i])
            elif model_types[i] == 'ureseffnetL':
                model = UResEfficientNetV2B0L(out_channels=out_channels[i])
            model.load_weights(model_paths[i])

            with tf.device("/GPU:0"):
                output += model.predict(data, verbose=0)[:, :, :, -1]
        
        output /= len(model_paths)

        if args.save_masks:
            np.save(os.path.join(args.logdir, "test_masks"), (output.numpy() >= 0.5).astype(np.uint8))

        iou = CAGS.evaluate_segmentation(cags.dev, output)
        tf.print("IoU on dev data: ", iou)

    else:

        data = cags.test.map(lambda x: x["image"])
        data = data.batch(args.batch_size)

        output = tf.zeros((612, 224, 224))

        for i in tqdm(range(len(model_paths))):
            # load model
            if model_types[i] == 'effunet':
                model = EfficientUNetV2B0(out_channels=out_channels[i])
            elif model_types[i] == 'ureseffnet':
                model = UResEfficientNetV2B0(out_channels=out_channels[i])
            elif model_types[i] == 'ureseffnetL':
                model = UResEfficientNetV2B0L(out_channels=out_channels[i])
            model.load_weights(model_paths[i])

            with tf.device("/GPU:0"):
                output += model.predict(data, verbose=0)[:, :, :, -1]
        
        output /= len(model_paths)

        if args.save_masks:
            np.save(os.path.join(args.logdir, "test_masks"), (output.numpy() >= 0.5).astype(np.uint8))
        
        with open(os.path.join(args.logdir, "cags_segmentation.txt"), "w", encoding="utf-8") as predictions_file:

            for mask in output:
                zeros, ones, runs = 0, 0, []
                for pixel in np.reshape(mask >= 0.5, [-1]):
                    if pixel:
                        if zeros or (not zeros and not ones):
                            runs.append(zeros)
                            zeros = 0
                        ones += 1
                    else:
                        if ones:
                            runs.append(ones)
                            ones = 0
                        zeros += 1
                runs.append(zeros + ones)
                print(*runs, file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)