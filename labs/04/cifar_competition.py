#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from cifar10 import CIFAR10

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--augment", default=False, action="store_true", help="Augmentation")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=12, type=int, help="Maximum number of threads to use.")


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.debug:
        tf.config.run_functions_eagerly(True)
        tf.data.experimental.enable_debug_mode()

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load data
    cifar = CIFAR10()

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

    train = tf.data.Dataset.from_tensor_slices((cifar.train.data["images"], cifar.train.data["labels"]))
    dev = tf.data.Dataset.from_tensor_slices((cifar.dev.data["images"], cifar.dev.data["labels"]))
    dev = dev.map(image_to_float)
    dev = dev.batch(args.batch_size)
    dev = dev.prefetch(tf.data.AUTOTUNE)

    train = train.shuffle(5000, seed=args.seed)
    train = train.map(image_to_float)

    if args.augment:
        train = train.map(train_augment_tf_image)

    train = train.batch(args.batch_size)
    train = train.prefetch(tf.data.AUTOTUNE)

    tst = tf.data.Dataset.from_tensor_slices((cifar.test.data["images"]))
    tst = tst.map(lambda x: tf.image.convert_image_dtype(x, tf.float32))
    tst = tst.batch(args.batch_size)

    # TODO: Create the model and train it
    # Create the model
    inputs = tf.keras.layers.Input(shape=[CIFAR10.H, CIFAR10.W, CIFAR10.C])
    hidden = tf.keras.layers.Conv2D(32, 3, 1, "same", activation=None)(inputs)
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.activations.relu(hidden)
    hidden = tf.keras.layers.Conv2D(32, 3, 2, "same", activation=None)(hidden)
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.activations.relu(hidden)
    hidden = tf.keras.layers.Conv2D(64, 3, 1, "same", activation=None)(hidden)
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.activations.relu(hidden)
    hidden = tf.keras.layers.Conv2D(64, 3, 2, "same", activation=None)(hidden)
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.activations.relu(hidden)
    hidden = tf.keras.layers.Conv2D(128, 3, 1, "same", activation=None)(hidden)
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.activations.relu(hidden)
    hidden = tf.keras.layers.Conv2D(128, 3, 2, "same", activation=None)(hidden)
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.activations.relu(hidden)
    hidden = tf.keras.layers.Flatten()(hidden)
    hidden = tf.keras.layers.Dense(32, activation=tf.nn.relu)(hidden)
    outputs = tf.keras.layers.Dense(len(CIFAR10.LABELS), activation=tf.nn.softmax)(hidden)

    # Compile the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.optimizers.Adam(jit_compile=False),
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)

    logs = model.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[tb_callback])

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cifar_competition_test.txt"), "w", encoding="utf-8") as predictions_file:
        for probs in model.predict(tst):
            print(np.argmax(probs), file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
