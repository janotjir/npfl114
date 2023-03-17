#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

from cifar10 import CIFAR10

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--augment", default=True, action="store_true", help="Augmentation")
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=50, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=16, type=int, help="Maximum number of threads to use.")

print(tf.config.get_visible_devices())


class Conv_pipeline(tf.keras.Sequential):
    def __init__(self, channels):
        super(Conv_pipeline, self).__init__()
        self.add(tf.keras.layers.Conv2D(channels, 3, 1, "same", activation=None))
        self.add(tf.keras.layers.BatchNormalization())
        self.add(tf.keras.layers.ReLU())
        self.add(tf.keras.layers.MaxPool2D((2,2)))


def model_builder(hp):
        model = keras.Sequential()

        # Tune the number of units in the first Dense layer
        # Choose an optimal value between 32-512
        layers = hp.Choice('layers', values=[2, 3, 4])
        start_channels = hp.Choice('s_channels', values=[8, 16, 24, 32, 64])
        dense = hp.Choice('dense', values=[32, 64, 128, 256, 512])
        lr = hp.Choice('lr', values=[0.001, 0.0001])
        decay = hp.Choice('decay', values=[0.0001, 0.00001])

        for i in range(layers):
            model.add(Conv_pipeline(start_channels*(i+1)))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(dense, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(len(CIFAR10.LABELS), activation=tf.nn.softmax))

        model.compile(optimizer=keras.optimizers.experimental.AdamW(learning_rate=lr, weight_decay=decay, jit_compile=False),
                        loss=tf.losses.SparseCategoricalCrossentropy(),
                        metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
                        )
        return model


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
    tuner = kt.Hyperband(model_builder, 'val_accuracy', max_epochs=args.epochs)

    tuner.search(train, epochs=args.epochs, validation_data=dev)

    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
    print(best_hps)
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(train, epochs=50, validation_data=dev)

    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))

    best_model = tuner.hypermodel.build(best_hps)
    best_model.fit(train, epochs=best_epoch, validation_data=dev)

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cifar_competition_test.txt"), "w", encoding="utf-8") as predictions_file:
        for probs in best_model.predict(tst):
            print(np.argmax(probs), file=predictions_file)

    best_model.save('the_best_fkin_model.h5', include_optimizer=False)
    eval_result = best_model.evaluate(dev)
    print(eval_result)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)