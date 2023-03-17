#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from uppercase_data import UppercaseData

# TODO: Set reasonable values for the hyperparameters, especially for
# `alphabet_size`, `batch_size`, `epochs`, and `windows`.
# Also, you can set the number of the threads 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--alphabet_size", default=256, type=int, help="If given, use this many most frequent chars.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=50, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--window", default=10, type=int, help="Window size to use.")
parser.add_argument("--model", default="", type=str, help="Output model name.")
parser.add_argument("--model_path", default=None, type=str, help="Path to trained model")

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

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load data
    uppercase_data = UppercaseData(args.window, args.alphabet_size)

    # TODO: Implement a suitable model, optionally including regularization, select
    # good hyperparameters and train the model.
    #
    # The inputs are _windows_ of fixed size (`args.window` characters on left,
    # the character in question, and `args.window` characters on right), where
    # each character is represented by a `tf.int32` index. To suitably represent
    # the characters, you can:
    # - Convert the character indices into _one-hot encoding_. There is no
    #   explicit Keras layer, but you can
    #   - use a Lambda layer which can encompass any function:
    #       tf.keras.Sequential([
    #         tf.keras.layers.Input(shape=[2 * args.window + 1], dtype=tf.int32),
    #         tf.keras.layers.Lambda(lambda x: tf.one_hot(x, len(uppercase_data.train.alphabet))),
    #         ...
    #       ])
    #   - or use Functional API and then any TF function can be used
    #     as a Keras layer:
    #       inputs = tf.keras.layers.Input(shape=[2 * args.window + 1], dtype=tf.int32)
    #       encoded = tf.one_hot(inputs, len(uppercase_data.train.alphabet))
    #   You can then flatten the one-hot encoded windows and follow with a dense layer.
    # - Alternatively, you can use `tf.keras.layers.Embedding` (which is an efficient
    #   implementation of one-hot encoding followed by a Dense layer) and flatten afterwards.
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input([2*args.window+1]))
    model.add(tf.keras.layers.Embedding(len(uppercase_data.train.alphabet), 128, input_length=2*args.window+1))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))


    model.compile(
        optimizer=tf.keras.optimizers.experimental.AdamW(learning_rate=0.001, weight_decay=0.005),
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.metrics.SparseCategoricalAccuracy("accuracy")],
    )

    #tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1)
    
    def save_model(epoch, logs):
        model.save(os.path.join(args.logdir, f"{args.model}_ep{epoch+1}.h5"), include_optimizer=False)

    if args.model_path is None:
        with tf.device("/GPU:0"):
            model.fit(
                uppercase_data.train.data["windows"], uppercase_data.train.data["labels"],
                batch_size=args.batch_size, epochs=args.epochs,
                validation_data=(uppercase_data.dev.data["windows"], uppercase_data.dev.data["labels"]),
                callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=save_model)],
            )
    else:
        model = tf.keras.models.load_model(args.model_path)

    # TODO: Generate correctly capitalized test set.
    # Use `uppercase_data.test.text` as input, capitalize suitable characters,
    # and write the result to predictions_file (which is
    # `uppercase_test.txt` in the `args.logdir` directory).

    def selective_upper(index, char, labels):
        if labels[index] > 0:
            char = char.upper()
        return char

    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "uppercase_test.txt"), "w", encoding="utf-8") as predictions_file:
        ch_list = [*uppercase_data.test.text]
        labels = model.predict(uppercase_data.test.data['windows'])
        labels = tf.math.argmax(labels, axis=1)
        u_list = [selective_upper(index, char, labels) for index, char in enumerate(ch_list)]
        u_text = "".join(u_list)
        predictions_file.write(u_text)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)