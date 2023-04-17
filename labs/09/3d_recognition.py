#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from modelnet import ModelNet

# Team members:
# 4c2c10df-00be-4008-8e01-1526b9225726
# dc535248-fa6c-4987-b49f-25b6ede7c87d

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
parser.add_argument("--modelnet", default=32, type=int, help="ModelNet dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

parser.add_argument("--test", default=False, action="store_true", help="Load model and only annotate test data")
parser.add_argument("--eval", default=False, action="store_true", help="Load model and only annotate dev data")
parser.add_argument("--model", default="", type=str, help="Model path")


class CNN3D(tf.keras.Model):
    def __init__(self, args, num_steps=None, k=2, N=1, mega_skip=False):
        inputs = tf.keras.layers.Input(shape=[args.modelnet, args.modelnet, args.modelnet, ModelNet.C], dtype=tf.float32)

        _hidden = tf.keras.layers.Conv3D(16, kernel_size=3, padding='same', activation=None, use_bias=False)(inputs)
        _hidden = tf.keras.layers.BatchNormalization()(_hidden)
        _hidden = tf.keras.layers.Activation('relu')(_hidden)

        hidden = self._block(_hidden, 16*k, stride=1)
        for i in range(N):
            hidden = self._identity_block(hidden, 16*k)

        hidden = self._block(hidden, 32*k, stride=2)
        for i in range(N):
            hidden = self._identity_block(hidden, 32*k)

        hidden = self._block(hidden, 64*k, stride=2)
        for i in range(N):
            hidden = self._identity_block(hidden, 64*k)

        if mega_skip:
            size = args.modelnet//4
            _hidden = tf.keras.layers.AveragePooling3D((size, size, size), strides=4, padding='same')(_hidden)
            _hidden = tf.keras.layers.Conv2D(64*k, kernel_size=1, strides=1, padding='same', activation=None, use_bias=False)(_hidden)
            _hidden = tf.keras.layers.BatchNormalization()(_hidden)

            hidden = tf.keras.layers.add([hidden, _hidden])
            hidden = tf.keras.layers.Activation('relu')(hidden)

        hidden = tf.keras.layers.Lambda(lambda z: tf.keras.backend.mean(z, [1, 2, 3]), name='reduce_mean')(hidden)
        outputs = tf.keras.layers.Dense(len(ModelNet.LABELS), activation=tf.nn.softmax)(hidden)

        super().__init__(inputs=inputs, outputs=outputs)

        if num_steps is not None:
            lr = tf.keras.optimizers.schedules.CosineDecay(1e-4, num_steps * args.epochs, alpha=1e-6)
        else:
            lr = 1e-3
        self.compile(
            optimizer=tf.optimizers.experimental.AdamW(clipnorm=1.0, learning_rate=lr, jit_compile=False),
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )

    def _identity_block(self, input, filters):
        x = tf.keras.layers.Conv3D(filters, kernel_size=3, padding='same', activation=None, use_bias=False)(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv3D(filters, kernel_size=3, padding='same', activation=None, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.add([x, input])
        x = tf.keras.layers.Activation('relu')(x)

        return x

    def _block(self, input, filters, stride=2):
        x = tf.keras.layers.Conv3D(filters, kernel_size=3, strides=1, padding='same', activation=None, use_bias=False)(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv3D(filters, kernel_size=3, strides=stride, padding='same', activation=None, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        input = tf.keras.layers.AveragePooling3D((2, 2, 2), strides=stride, padding='same')(input)
        input = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=1, padding='same', activation=None, use_bias=False)(input)
        input = tf.keras.layers.BatchNormalization()(input)

        x = tf.keras.layers.add([x, input])
        x = tf.keras.layers.Activation('relu')(x)

        return x


def main(args: argparse.Namespace) -> None:
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
    modelnet = ModelNet(args.modelnet)

    if not args.eval and not args.test:

        train = tf.data.Dataset.from_tensor_slices((modelnet.train.data["voxels"], modelnet.train.data["labels"]))
        train = train.shuffle(5000, seed=args.seed)
        train = train.batch(args.batch_size)
        train = train.prefetch(tf.data.AUTOTUNE)

        dev = tf.data.Dataset.from_tensor_slices((modelnet.dev.data["voxels"], modelnet.dev.data["labels"]))
        dev = dev.batch(args.batch_size)
        dev = dev.prefetch(tf.data.AUTOTUNE)

        # TODO: Create the model and train it
        model = CNN3D(args, len(train))

        tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)
        def save_model(epoch, logs):
            model.save(os.path.join(args.logdir, f"ep{epoch+1}.h5"), include_optimizer=False)

        model.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[tb_callback, tf.keras.callbacks.LambdaCallback(on_epoch_end=save_model)])

    else:
        model = CNN3D(args)
        model.load_weights(args.model)
        args.logdir = "/".join(args.model.split("/")[:-1])

    if args.eval:
        test = tf.data.Dataset.from_tensor_slices((modelnet.dev.data["voxels"], modelnet.dev.data["labels"]))
        filename = "3d_recognition_val.txt"
    else:
        test = tf.data.Dataset.from_tensor_slices((modelnet.test.data["voxels"], modelnet.test.data["labels"]))
        filename = "3d_recognition.txt"
    
    test = test.batch(args.batch_size)
    
    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, filename), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the probabilities on the test set
        test_probabilities = model.predict(test)

        for probs in test_probabilities:
            print(np.argmax(probs), file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
