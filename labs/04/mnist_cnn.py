#!/usr/bin/env python3
import argparse
import datetime
import os
import re
from typing import Dict
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--cnn", default=None, type=str, help="CNN architecture.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.

# Team members:
# 4c2c10df-00be-4008-8e01-1526b9225726
# dc535248-fa6c-4987-b49f-25b6ede7c87d


class Model(tf.keras.Model):
    def __init__(self, args: argparse.Namespace) -> None:
        # TODO: Create the model. The template uses the functional API, but
        # feel free to use subclassing if you want.
        inputs = tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C])

        # TODO: Add CNN layers specified by `args.cnn`, which contains
        # a comma-separated list of the following layers:
        # - `C-filters-kernel_size-stride-padding`: Add a convolutional layer with ReLU
        #   activation and specified number of filters, kernel size, stride and padding.
        # - `CB-filters-kernel_size-stride-padding`: Same as `C`, but use batch normalization.
        #   In detail, start with a convolutional layer without bias and activation,
        #   then add a batch normalization layer, and finally the ReLU activation.
        # - `M-pool_size-stride`: Add max pooling with specified size and stride, using
        #   the default "valid" padding.
        # - `R-[layers]`: Add a residual connection. The `layers` contain a specification
        #   of at least one convolutional layer (but not a recursive residual connection `R`).
        #   The input to the `R` layer should be processed sequentially by `layers`, and the
        #   produced output (after the ReLU nonlinearity of the last layer) should be added
        #   to the input (of this `R` layer).
        # - `F`: Flatten inputs. Must appear exactly once in the architecture.
        # - `H-hidden_layer_size`: Add a dense layer with ReLU activation and the specified size.
        # - `D-dropout_rate`: Apply dropout with the given dropout rate.
        # You can assume the resulting network is valid; it is fine to crash if it is not.
        #
        # Produce the results in the variable `hidden`.

        hidden = inputs
        residual = None
        residual_active = False

        layer_params = args.cnn.split(',')
        for layer_param in layer_params:
            layer_param = layer_param.split('-')
            if layer_param[0] == 'C':
                residual_terminal = False
                if residual_active and "]" in layer_param[-1]:
                    layer_param[-1] = layer_param[-1][:-1]
                    residual_terminal = True
                hidden = tf.keras.layers.Conv2D(int(layer_param[1]), int(layer_param[2]), strides=(int(layer_param[3]), int(layer_param[3])), 
                padding=layer_param[4], activation='relu')(hidden)
                if residual_terminal:
                    hidden = hidden + residual
                    residual = None
                    residual_active = False
            
            elif layer_param[0] == 'CB':
                residual_terminal = False
                if residual_active and "]" in layer_param[-1]:
                    layer_param[-1] = layer_param[-1][:-1]
                    residual_terminal = True
                hidden = tf.keras.layers.Conv2D(int(layer_param[1]), int(layer_param[2]), strides=(int(layer_param[3]), int(layer_param[3])), 
                padding=layer_param[4], activation=None, use_bias=False)(hidden)
                hidden = tf.keras.layers.BatchNormalization()(hidden)
                hidden = tf.keras.layers.Activation('relu')(hidden)
                if residual_terminal:
                    hidden = hidden + residual
                    residual = None
                    residual_active = False
            
            elif layer_param[0] == 'M':
                hidden = tf.keras.layers.MaxPool2D(pool_size=(int(layer_param[1]), int(layer_param[1])), strides=(int(layer_param[2]), int(layer_param[2])))(hidden)
            
            elif layer_param[0] == 'R':
                residual = tf.identity(hidden)
                layer_param[1] = layer_param[1][1:]
                if "]" not in layer_param[-1]:
                    residual_active = True
                else:
                    layer_param[-1] = layer_param[-1][:-1]
                if layer_param[1] == 'C':
                    hidden = tf.keras.layers.Conv2D(int(layer_param[2]), int(layer_param[3]), strides=(int(layer_param[4]), int(layer_param[4])), 
                    padding=layer_param[5], activation='relu')(hidden)
                elif layer_param[1] == 'CB':
                    hidden = tf.keras.layers.Conv2D(int(layer_param[2]), int(layer_param[3]), strides=(int(layer_param[4]), int(layer_param[4])), 
                    padding=layer_param[5], activation=None, use_bias=False)(hidden)
                    hidden = tf.keras.layers.BatchNormalization()(hidden)
                    hidden = tf.keras.layers.Activation('relu')(hidden)
                if not residual_active:
                    hidden = hidden + residual
                    residual = None
            
            elif layer_param[0] == 'F':
                hidden = tf.keras.layers.Flatten()(hidden)
            
            elif layer_param[0] == 'H':
                hidden = tf.keras.layers.Dense(int(layer_param[1]), activation='relu')(hidden)
            
            elif layer_param[0] == 'D':
                hidden = tf.keras.layers.Dropout(float(layer_param[1]))(hidden)

        # Add the final output layer
        outputs = tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax)(hidden)

        super().__init__(inputs=inputs, outputs=outputs)
        self.compile(
            optimizer=tf.optimizers.Adam(jit_compile=False),
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)


def main(args: argparse.Namespace) -> Dict[str, float]:
    # Set the random seed and the number of threads.
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.debug:
        tf.config.run_functions_eagerly(True)
        # tf.data.experimental.enable_debug_mode()

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the data
    mnist = MNIST()

    # Create the model and train it
    model = Model(args)

    logs = model.fit(
        mnist.train.data["images"], mnist.train.data["labels"],
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(mnist.dev.data["images"], mnist.dev.data["labels"]),
        callbacks=[model.tb_callback],
    )

    # Return development metrics for ReCodEx to validate.
    return {metric: values[-1] for metric, values in logs.history.items() if metric.startswith("val_")}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
