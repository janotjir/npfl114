#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from modelnet import ModelNet

import matplotlib.pyplot as plt
from tqdm import tqdm

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=12, type=int, help="Maximum number of threads to use.")
parser.add_argument("--test", default=False, action="store_true", help="Run model on test dataset")


# Team members:
# 4c2c10df-00be-4008-8e01-1526b9225726
# dc535248-fa6c-4987-b49f-25b6ede7c87d


class CNN3D(tf.keras.Model):
    def __init__(self, args, num_steps=None, k=2, N=1, mega_skip=False, modelnet=20):
        inputs = tf.keras.layers.Input(shape=[modelnet, modelnet, modelnet, ModelNet.C], dtype=tf.float32)

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
            size = modelnet//4
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
    # tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.debug:
        tf.config.run_functions_eagerly(True)
        tf.data.experimental.enable_debug_mode()

    args.logdir = "ensemble_3d_recog_results"
    os.makedirs(args.logdir, exist_ok=True)

    '''
    3d_recognition.py-2023-04-17_010614-bs=32,d=False,e=10,e=False,m=,m=20,s=42,t=False,t=1/ep8.h5 93.77 20 N=1 False
    3d_recognition.py-2023-04-17_010850-bs=32,d=False,e=10,e=False,m=,m=20,s=42,t=False,t=1/ep9.h5 94.14 20 N=1 False
    3d_recognition.py-2023-04-17_011124-bs=32,d=False,e=10,e=False,m=,m=20,s=42,t=False,t=1/ep10.h5 93.41 20 N=2 False
    3d_recognition.py-2023-04-17_011525-bs=32,d=False,e=10,e=False,m=,m=32,s=42,t=False,t=1/ep7.h5 94.51 32 N=1 False
    3d_recognition.py-2023-04-17_012337-bs=32,d=False,e=20,e=False,m=,m=32,s=42,t=False,t=1/ep14.h5 95.60 32 N=1 False
    3d_recognition.py-2023-04-17_123755-bs=32,d=False,e=20,e=False,m=,m=32,s=42,t=False,t=1/ep18.h5 95.24 32 N=1 True
    '''

    # 94.87
    '''model_paths = ["logs/3d_recognition.py-2023-04-17_010614-bs=32,d=False,e=10,e=False,m=,m=20,s=42,t=False,t=1/ep8.h5",
    "logs/3d_recognition.py-2023-04-17_010850-bs=32,d=False,e=10,e=False,m=,m=20,s=42,t=False,t=1/ep9.h5",
    "logs/3d_recognition.py-2023-04-17_011124-bs=32,d=False,e=10,e=False,m=,m=20,s=42,t=False,t=1/ep10.h5",
    "logs/3d_recognition.py-2023-04-17_011525-bs=32,d=False,e=10,e=False,m=,m=32,s=42,t=False,t=1/ep7.h5",
    "logs/3d_recognition.py-2023-04-17_012337-bs=32,d=False,e=20,e=False,m=,m=32,s=42,t=False,t=1/ep14.h5",
    "logs/3d_recognition.py-2023-04-17_123755-bs=32,d=False,e=20,e=False,m=,m=32,s=42,t=False,t=1/ep18.h5"]
    Ns = [1, 1, 2, 1, 1, 1]
    mega_skips = [False, False, False, False, False, True]
    modelnets = [20, 20, 20, 32, 32, 32]'''

    # 95.24
    '''model_paths = ["logs/3d_recognition.py-2023-04-17_011525-bs=32,d=False,e=10,e=False,m=,m=32,s=42,t=False,t=1/ep7.h5",
    "logs/3d_recognition.py-2023-04-17_012337-bs=32,d=False,e=20,e=False,m=,m=32,s=42,t=False,t=1/ep14.h5",
    "logs/3d_recognition.py-2023-04-17_123755-bs=32,d=False,e=20,e=False,m=,m=32,s=42,t=False,t=1/ep18.h5"]
    Ns = [1, 1, 1]
    mega_skips = [False, False, True]
    modelnets = [32, 32, 32]'''

    # 95.24
    '''model_paths = ["logs/3d_recognition.py-2023-04-17_012337-bs=32,d=False,e=20,e=False,m=,m=32,s=42,t=False,t=1/ep14.h5",
    "logs/3d_recognition.py-2023-04-17_123755-bs=32,d=False,e=20,e=False,m=,m=32,s=42,t=False,t=1/ep18.h5"]
    Ns = [1, 1]
    mega_skips = [False, True]
    modelnets = [32, 32]'''

    # 95.24
    model_paths = ["logs/3d_recognition.py-2023-04-17_010850-bs=32,d=False,e=10,e=False,m=,m=20,s=42,t=False,t=1/ep9.h5",
    "logs/3d_recognition.py-2023-04-17_011525-bs=32,d=False,e=10,e=False,m=,m=32,s=42,t=False,t=1/ep7.h5",
    "logs/3d_recognition.py-2023-04-17_012337-bs=32,d=False,e=20,e=False,m=,m=32,s=42,t=False,t=1/ep14.h5",
    "logs/3d_recognition.py-2023-04-17_123755-bs=32,d=False,e=20,e=False,m=,m=32,s=42,t=False,t=1/ep18.h5"]
    Ns = [1, 1, 1, 1]
    mega_skips = [False, False, False, True]
    modelnets = [20, 32, 32, 32]

    if not args.test:
        filename = "3d_recognition_val.txt"
        output = tf.zeros([273, len(ModelNet.LABELS)])
    else:
        filename = "3d_recognition.txt"
        output = tf.zeros([908, len(ModelNet.LABELS)])
    
    for i in tqdm(range(len(model_paths))):
        model = CNN3D(args, N=Ns[i], mega_skip=mega_skips[i], modelnet=modelnets[i])
        model.load_weights(model_paths[i])

        modelnet = ModelNet(modelnets[i])
        if not args.test:
            test = tf.data.Dataset.from_tensor_slices((modelnet.dev.data["voxels"], modelnet.dev.data["labels"]))
        else:
            test = tf.data.Dataset.from_tensor_slices((modelnet.test.data["voxels"], modelnet.test.data["labels"]))
        test = test.batch(args.batch_size)

        with tf.device("/GPU:0"):
            output += model.predict(test, verbose=0)
    
    output /= len(model_paths)

    with open(os.path.join(args.logdir, filename), "w", encoding="utf-8") as predictions_file:
        for probs in output:
            print(np.argmax(probs), file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)