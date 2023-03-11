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

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--augment", default=False, action="store_true", help="Augmentation")
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=12, type=int, help="Maximum number of threads to use.")
parser.add_argument("--model", default='xresnet18', type=str, choices=['xresnet18', 'resnet34', 'resnet50', 'resnet9'], help="Specify which model to train.")
parser.add_argument("--test", default=False, action="store_true", help="Run model on test dataset")
parser.add_argument("--model_path", default="", type=str, help="Specify model to trained model")


class ResNet50(tf.keras.Model):
    def __init__(self):
        inputs = tf.keras.layers.Input(shape=[CIFAR10.H, CIFAR10.W, CIFAR10.C])

        x = tf.keras.layers.Conv2D(64, 7, strides=2, padding='same', activation=None, use_bias=False)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        print(x.shape)

        x = tf.keras.layers.MaxPool2D((3,3), (2,2), padding='same')(x)

        print(x.shape)

        # 1st block
        x = self._conv_block(x, [64, 64, 256])
        x = self._identity_block(x, [64, 64, 256])
        x = self._identity_block(x, [64, 64, 256])

        print(x.shape)

        # 2nd block
        x = self._conv_block(x, [128, 128, 512])
        x = self._identity_block(x, [128, 128, 512])
        x = self._identity_block(x, [128, 128, 512])
        x = self._identity_block(x, [128, 128, 512])

        print(x.shape)

        # 3rd block
        x = self._conv_block(x, [256, 256, 1024])
        x = self._identity_block(x, [256, 256, 1024])
        x = self._identity_block(x, [256, 256, 1024])
        x = self._identity_block(x, [256, 256, 1024])
        x = self._identity_block(x, [256, 256, 1024])
        x = self._identity_block(x, [256, 256, 1024])

        print(x.shape)

        # 4th block
        x = self._conv_block(x, [512, 512, 2018])
        x = self._identity_block(x, [512, 512, 2018])
        x = self._identity_block(x, [512, 512, 2018])

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

    def _identity_block(self, input, filters):
        x = tf.keras.layers.Conv2D(filters[0], kernel_size=1, strides=1, padding='same', activation=None, use_bias=False)(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv2D(filters[1], kernel_size=3, strides=1, padding='same', activation=None, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv2D(filters[2], kernel_size=1, strides=1, padding='same', activation=None, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.add([x, input])
        x = tf.keras.layers.Activation('relu')(x)
        
        return x

    def _conv_block(self, input, filters, stride=1):
        x = tf.keras.layers.Conv2D(filters[0], kernel_size=1, strides=1, padding='same', activation=None, use_bias=False)(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv2D(filters[1], kernel_size=3, strides=1, padding='same', activation=None, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv2D(filters[2], kernel_size=1, strides=1, padding='same', activation=None, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        input = tf.keras.layers.Conv2D(filters[2], kernel_size=1, strides=stride, padding='same', activation=None, use_bias=False)(input)
        input = tf.keras.layers.BatchNormalization()(input)

        x = tf.keras.layers.add([x, input])
        x = tf.keras.layers.Activation('relu')(x)
        
        return x


class ResNet34(tf.keras.Model):
    def __init__(self):
        inputs = tf.keras.layers.Input(shape=[CIFAR10.H, CIFAR10.W, CIFAR10.C])

        x = tf.keras.layers.Conv2D(64, 7, strides=2, padding='same', activation=None, use_bias=False)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        print(x.shape)

        #x = tf.keras.layers.MaxPool2D((3,3), (2,2), padding='same')(x)
        #print(x.shape)

        # 1st block
        x = self._conv_block(x, [64, 64])
        x = self._identity_block(x, [64, 64])
        x = self._identity_block(x, [64, 64])
        print(x.shape)

        # 2nd block
        x = self._conv_block(x, [128, 128])
        x = self._identity_block(x, [128, 128])
        x = self._identity_block(x, [128, 128])
        x = self._identity_block(x, [128, 128])
        print(x.shape)

        # 3rd block
        x = self._conv_block(x, [256, 256])
        x = self._identity_block(x, [256, 256])
        x = self._identity_block(x, [256, 256])
        x = self._identity_block(x, [256, 256])
        x = self._identity_block(x, [256, 256])
        x = self._identity_block(x, [256, 256])
        print(x.shape)

        # 4th block
        x = self._conv_block(x, [512, 512])
        x = self._identity_block(x, [512, 512])
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

    def _identity_block(self, input, filters):
        x = tf.keras.layers.Conv2D(filters[0], kernel_size=3, strides=1, padding='same', activation=None, use_bias=False)(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv2D(filters[1], kernel_size=3, strides=1, padding='same', activation=None, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.add([x, input])
        x = tf.keras.layers.Activation('relu')(x)
        
        return x

    def _conv_block(self, input, filters):
        x = tf.keras.layers.Conv2D(filters[0], kernel_size=3, strides=2, padding='same', activation=None, use_bias=False)(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv2D(filters[1], kernel_size=3, strides=1, padding='same', activation=None, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        input = tf.keras.layers.Conv2D(filters[1], kernel_size=1, strides=2, padding='same', activation=None, use_bias=False)(input)
        input = tf.keras.layers.BatchNormalization()(input)

        x = tf.keras.layers.add([x, input])
        x = tf.keras.layers.Activation('relu')(x)
        
        return x


class XResNet18(tf.keras.Model):
    def __init__(self):
        inputs = tf.keras.layers.Input(shape=[CIFAR10.H, CIFAR10.W, CIFAR10.C])

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
    def __init__(self):
        inputs = tf.keras.layers.Input(shape=[CIFAR10.H, CIFAR10.W, CIFAR10.C])

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
        self.compile(
            optimizer=tf.optimizers.experimental.AdamW(learning_rate=0.001, weight_decay= 0.00001, jit_compile=False),
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

    if not args.test:

        train = tf.data.Dataset.from_tensor_slices((cifar.train.data["images"], cifar.train.data["labels"]))
        dev = tf.data.Dataset.from_tensor_slices((cifar.dev.data["images"], cifar.dev.data["labels"]))
        dev = dev.map(image_to_float)
        dev = dev.batch(args.batch_size)
        dev = dev.prefetch(tf.data.AUTOTUNE)

        train = train.shuffle(5000, seed=args.seed)
        train = train.map(image_to_float)

        if args.augment:
            train = train.map(train_augment_tf_image)
        
        if args.normalize:
            train = train.map(normalise)

        train = train.batch(args.batch_size)
        train = train.prefetch(tf.data.AUTOTUNE)

        # Create the model
        if args.model == 'xresnet18':
            model = XResNet18()
        elif args.model == 'resnet34':
            model = ResNet34()
        elif args.model == 'resnet50':
            model = ResNet50()
        elif args.model == 'resnet9':
            model = ResNet9()

        def save_model(epoch, logs):
            if epoch+1 % 5 == 0:
                model.save(os.path.join(args.logdir, f"{args.model}_ep{epoch+1}.h5"), include_optimizer=False)

        with tf.device("/GPU:0"):
            model.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=save_model), model.tb_callback])

    else:
        # load trained model
        if args.model_path == "":
            print("Specify path to trained model")
            exit()
        
        if args.model == 'xresnet18':
            model = XResNet18()
        elif args.model == 'resnet34':
            model = ResNet34()
        elif args.model == 'resnet50':
            model = ResNet50()
        elif args.model == 'resnet9':
            model = ResNet9()

        model.load_weights(args.model_path)


    tst = tf.data.Dataset.from_tensor_slices((cifar.test.data["images"]))
    tst = tst.map(lambda x: tf.image.convert_image_dtype(x, tf.float32))
    tst = tst.batch(args.batch_size)

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    if args.test: 
        args.logdir = ""
    else:
        os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cifar_competition_test.txt"), "w", encoding="utf-8") as predictions_file:
        for probs in model.predict(tst):
            print(np.argmax(probs), file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
