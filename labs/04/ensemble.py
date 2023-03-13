#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from cifar10 import CIFAR10

import matplotlib.pyplot as plt
from tqdm import tqdm

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--augment", default=False, action="store_true", help="Augmentation")
parser.add_argument("--normalize", default=False, action="store_true", help="Per-channel normalization")
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=12, type=int, help="Maximum number of threads to use.")
parser.add_argument("--test", default=False, action="store_true", help="Run model on test dataset")


# Team members:
# 4c2c10df-00be-4008-8e01-1526b9225726
# dc535248-fa6c-4987-b49f-25b6ede7c87d


class XResNet18(tf.keras.Model):
    def __init__(self, normalize):
        inputs = tf.keras.layers.Input(shape=[CIFAR10.H, CIFAR10.W, CIFAR10.C])

        if normalize:
          mean = [0.4914, 0.4822, 0.4465]
          std = [0.2023, 0.1994, 0.2010]
          inputs = tf.keras.layers.Lambda(lambda x: (x - mean) / std)(inputs)

        # ResNet-C improvement - input stem
        x = tf.keras.layers.Conv2D(32, 3, strides=2, padding='same', activation=None, use_bias=False)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        print(x.shape)

        x = tf.keras.layers.Conv2D(64, 3, strides=1, padding='same', activation=None, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        print(x.shape)

        x = tf.keras.layers.Conv2D(64, 3, strides=1, padding='same', activation=None, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        print(x.shape)

        # 1st block
        x = self._identity_block(x, [64, 64])
        print(x.shape)
        x = self._identity_block(x, [64, 64])
        print(x.shape)

        # 2nd block
        x = self._block(x, [128, 128])
        print(x.shape)
        x = self._identity_block(x, [128, 128])
        print(x.shape)

        # 3rd block
        x = self._block(x, [256, 256])
        print(x.shape)
        x = self._identity_block(x, [256, 256])
        print(x.shape)

        # 4th block
        x = self._block(x, [512, 512])
        print(x.shape)
        x = self._identity_block(x, [512, 512])
        print(x.shape)

        # final
        x = tf.keras.layers.Lambda(lambda z: tf.keras.backend.mean(z, [1, 2]), name='reduce_mean')(x)
        print(x.shape)
        outputs = tf.keras.layers.Dense(len(CIFAR10.LABELS), activation=tf.nn.softmax)(x)
        print(outputs.shape)

        super().__init__(inputs=inputs, outputs=outputs)
        self.compile(
            optimizer=tf.optimizers.experimental.AdamW(jit_compile=False),
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)

    def _identity_block(self, input, filters, stride=1):
        x = tf.keras.layers.Conv2D(filters[0], kernel_size=3, strides=stride, padding='same', activation=None, use_bias=False)(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv2D(filters[1], kernel_size=3, strides=1, padding='same', activation=None, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.add([x, input])
        x = tf.keras.layers.Activation('relu')(x)
        
        return x

    def _block(self, input, filters, stride=2):
        # ResNet-B improvement
        x = tf.keras.layers.Conv2D(filters[0], kernel_size=3, strides=1, padding='same', activation=None, use_bias=False)(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv2D(filters[1], kernel_size=3, strides=stride, padding='same', activation=None, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        # ResNet-D improvement
        input = tf.keras.layers.AveragePooling2D((2,2), strides=2)(input)
        input = tf.keras.layers.Conv2D(filters[1], kernel_size=1, strides=1, padding='same', activation=None, use_bias=False)(input)
        input = tf.keras.layers.BatchNormalization()(input)

        x = tf.keras.layers.add([x, input])
        x = tf.keras.layers.Activation('relu')(x)
        
        return x


class ResNet9(tf.keras.Model):
    def __init__(self, normalize):
        inputs = tf.keras.layers.Input(shape=[CIFAR10.H, CIFAR10.W, CIFAR10.C])

        if normalize:
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
            inputs = tf.keras.layers.Lambda(lambda x: (x - mean) / std)(inputs)

        x = tf.keras.layers.Conv2D(64, 3, strides=1, padding='same', activation=None, use_bias=False)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        print(x.shape)

        x = tf.keras.layers.Conv2D(128, 3, strides=1, padding='same', activation=None, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        print(x.shape)

        x = tf.keras.layers.MaxPool2D((2,2), strides=2)(x)
        print(x.shape)

        x = self._identity_block(x, [128, 128])
        print(x.shape)

        x = tf.keras.layers.Conv2D(256, 3, strides=1, padding='same', activation=None, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        print(x.shape)

        x = tf.keras.layers.MaxPool2D((2,2), strides=2)(x)
        print(x.shape)

        x = tf.keras.layers.Conv2D(256, 3, strides=1, padding='same', activation=None, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        print(x.shape)

        x = tf.keras.layers.MaxPool2D((2,2), strides=2)(x)
        print(x.shape)

        x = self._identity_block(x, [256, 256])
        print(x.shape)

        x = tf.keras.layers.MaxPool2D((2,2), strides=2)(x)
        print(x.shape)

        x = tf.keras.layers.Flatten()(x)
        print(x.shape)

        # final
        #x = tf.keras.layers.Lambda(lambda z: tf.keras.backend.mean(z, [1, 2]), name='reduce_mean')(x)
        #print(x.shape)
        outputs = tf.keras.layers.Dense(len(CIFAR10.LABELS), activation=tf.nn.softmax)(x)
        print(outputs.shape)

        super().__init__(inputs=inputs, outputs=outputs)
        '''clr = CyclicalLearningRate(initial_learning_rate=0.001, 
                                   maximal_learning_rate=0.01,
                                   scale_fn=lambda x: 1/(2.**(x-1)),
                                   step_size=2*88)'''
        clr = lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.001,
                                                                           decay_steps=1500,
                                                                           decay_rate=0.96,
                                                                           staircase=True)


        self.compile(
            #optimizer=tf.optimizers.experimental.AdamW(learning_rate=0.001, weight_decay= 0.0001, jit_compile=False),
            #optimizer=tf.optimizers.experimental.AdamW(learning_rate=0.001, weight_decay= 0.0001, clipvalue=0.1, jit_compile=False),
            optimizer=tf.optimizers.experimental.AdamW(learning_rate=clr, weight_decay= 0.0001, clipvalue=0.1, jit_compile=False),
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)

    def _identity_block(self, input, filters, stride=1):
        x = tf.keras.layers.Conv2D(filters[0], kernel_size=3, strides=stride, padding='same', activation=None, use_bias=False)(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv2D(filters[1], kernel_size=3, strides=1, padding='same', activation=None, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.add([x, input])
        x = tf.keras.layers.Activation('relu')(x)
        
        return x

    def _block(self, input, filters, stride=2):
        # ResNet-B improvement
        x = tf.keras.layers.Conv2D(filters[0], kernel_size=3, strides=1, padding='same', activation=None, use_bias=False)(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv2D(filters[1], kernel_size=3, strides=stride, padding='same', activation=None, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        # ResNet-D improvement
        input = tf.keras.layers.AveragePooling2D((2,2), strides=2)(input)
        input = tf.keras.layers.Conv2D(filters[1], kernel_size=1, strides=1, padding='same', activation=None, use_bias=False)(input)
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

    args.logdir = "ensemble_results"

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

    # 89.6
    model_types = ['xresnet18', 'resnet9', 'resnet9', 'xresnet18']
    normalize = [False, True, True, True]
    model_paths = ["logs/cifar_competition.py-2023-03-10_174055-a=True,bs=128,d=False,e=20,m=xresnet18,s=42,t=12/xresnet18_ep20.h5",
    "logs/resnet9_2/resnet9_ep16.h5",
    "logs/resnet9_4/resnet9_ep25.h5",
    "logs/xresnet18_3/xresnet18_ep31.h5"]

    # 92.96
    model_types = ['resnet9', 'resnet9']
    normalize = [True, True]
    model_paths = ["logs/resnet9_3/resnet9_ep40.h5",
    "logs/resnet9_5/resnet9_ep40.h5"]

    # 92.56
    '''model_types = ['xresnet18', 'resnet9', 'resnet9', 'xresnet18', 'resnet9', 'resnet9']
    normalize = [False, True, True, True, True, True]
    model_paths = ["logs/cifar_competition.py-2023-03-10_174055-a=True,bs=128,d=False,e=20,m=xresnet18,s=42,t=12/xresnet18_ep20.h5",
    "logs/resnet9_2/resnet9_ep16.h5",
    "logs/resnet9_4/resnet9_ep25.h5",
    "logs/xresnet18_3/xresnet18_ep31.h5",
    "logs/resnet9_3/resnet9_ep40.h5",
    "logs/resnet9_5/resnet9_ep40.h5"]'''

    # 93.02
    model_types = ['resnet9', 'resnet9', 'resnet9', 'resnet9']
    normalize = [True, True, True, True]
    model_paths = ["logs/resnet9_3/resnet9_ep39.h5",
    "logs/resnet9_3/resnet9_ep40.h5",
    "logs/resnet9_5/resnet9_ep37.h5",
    "logs/resnet9_5/resnet9_ep40.h5"]

    # 93.08
    model_types = ['resnet9', 'resnet9', 'resnet9', 'resnet9',  'resnet9', 'resnet9']
    normalize = [True, True, True, True, True, True]
    model_paths = ["logs/resnet9_3/resnet9_ep39.h5",
    "logs/resnet9_3/resnet9_ep40.h5",
    "logs/resnet9_3/resnet9_ep36.h5",
    "logs/resnet9_5/resnet9_ep37.h5",
    "logs/resnet9_5/resnet9_ep36.h5",
    "logs/resnet9_5/resnet9_ep40.h5"]


    if not args.test:
        dev = tf.data.Dataset.from_tensor_slices((cifar.dev.data["images"], cifar.dev.data["labels"]))
        dev = dev.map(image_to_float)
        dev = dev.batch(args.batch_size)
        dev = dev.prefetch(tf.data.AUTOTUNE)

        output = tf.zeros([cifar.dev.data["images"].shape[0], 10])

        for i in tqdm(range(len(model_paths))):
            # load model
            if model_types[i] == 'xresnet18':
                model = XResNet18(normalize[i])
            elif model_types[i] == 'resnet9':
                model = ResNet9(normalize[i])
            model.load_weights(model_paths[i])

            with tf.device("/GPU:0"):
                output += model.predict(dev, verbose=0)
        
        output /= len(model_paths)

        ensemble_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        ensemble_accuracy.update_state(cifar.dev.data["labels"], output)
        ensemble_accuracy = ensemble_accuracy.result().numpy()
        print(f"Ensemble accuracy on dev data: {ensemble_accuracy}")

    else:

        tst = tf.data.Dataset.from_tensor_slices((cifar.test.data["images"]))
        tst = tst.map(lambda x: tf.image.convert_image_dtype(x, tf.float32))
        tst = tst.batch(args.batch_size)

        output = tf.zeros([cifar.test.data["images"].shape[0], 10])

        for i in tqdm(range(len(model_paths))):
            # load model
            if model_types[i] == 'xresnet18':
                model = XResNet18(normalize[i])
            elif model_types[i] == 'resnet9':
                model = ResNet9(normalize[i])
            model.load_weights(model_paths[i])

            with tf.device("/GPU:0"):
                output += model.predict(tst, verbose=0)
        
        output /= len(model_paths)

        args.logdir = "ensemble_results"
        os.makedirs(args.logdir, exist_ok=True)
        with open(os.path.join(args.logdir, "cifar_competition_test.txt"), "w", encoding="utf-8") as predictions_file:
            for probs in output:
                print(np.argmax(probs), file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)