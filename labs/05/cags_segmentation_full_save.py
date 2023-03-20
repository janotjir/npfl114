#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from cags_dataset import CAGS

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of the threads 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--augment", default=False, action='store_true', help='Augment the training data')
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--test", default=False, action="store_true", help="Run model on test dataset")
parser.add_argument("--model_path", default="", type=str, help="Specify path to trained model")


# Team members:
# 4c2c10df-00be-4008-8e01-1526b9225726
# dc535248-fa6c-4987-b49f-25b6ede7c87d


class EfficientUNetV2B0(tf.keras.Model):
    def __init__(self):
        inputs = tf.keras.layers.Input(shape=[CAGS.H, CAGS.W, CAGS.C])

        # Load the EfficientNetV2-B0 model
        backbone = tf.keras.applications.EfficientNetV2B0(include_top=False)
        backbone = tf.keras.Model(
            inputs=backbone.input,
            outputs=[backbone.get_layer(layer).output for layer in [
                "top_activation", "block5e_add", "block3b_add", "block2b_add", "stem_activation"]]
        )
        backbone.trainable = False

        inputs = tf.keras.layers.Input(shape=[CAGS.H, CAGS.W, CAGS.C])
        bb_out = backbone(inputs)
        
        hidden = tf.keras.layers.Conv2D(640, 3, use_bias=False, padding='same')(bb_out[0])
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.keras.layers.Activation(tf.keras.activations.swish)(hidden)

        hidden = tf.keras.layers.Conv2DTranspose(320, 3, strides=2, use_bias=False, padding='same')(hidden)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.keras.layers.Concatenate(axis=-1)([hidden, bb_out[1]])

        hidden = tf.keras.layers.Conv2D(256, 3, use_bias=False, padding='same')(hidden)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.keras.layers.Activation(tf.keras.activations.swish)(hidden)

        hidden = tf.keras.layers.Conv2DTranspose(128, 3, strides=2, use_bias=False, padding='same')(hidden)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.keras.layers.Concatenate(axis=-1)([hidden, bb_out[2]])

        hidden = tf.keras.layers.Conv2D(128, 3, use_bias=False, padding='same')(hidden)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.keras.layers.Activation(tf.keras.activations.swish)(hidden)

        hidden = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, use_bias=False, padding='same')(hidden)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.keras.layers.Concatenate(axis=-1)([hidden, bb_out[3]])

        hidden = tf.keras.layers.Conv2D(64, 3, use_bias=False, padding='same')(hidden)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.keras.layers.Activation(tf.keras.activations.swish)(hidden)

        hidden = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, use_bias=False, padding='same')(hidden)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.keras.layers.Concatenate(axis=-1)([hidden, bb_out[4]])

        hidden = tf.keras.layers.Conv2D(32, 3, use_bias=False, padding='same')(hidden)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.keras.layers.Activation(tf.keras.activations.swish)(hidden)

        hidden = tf.keras.layers.Conv2DTranspose(16, 3, strides=2, use_bias=False, padding='same')(hidden)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.keras.layers.Activation(tf.keras.activations.swish)(hidden)

        hidden = tf.keras.layers.Conv2D(1, 1, use_bias=False, padding='same')(hidden)
        hidden = tf.keras.layers.Activation(tf.nn.sigmoid)(hidden)

        super().__init__(inputs=inputs, outputs=hidden)


def main(args: argparse.Namespace) -> None:
    # Fix random seeds and threads
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the data
    cags = CAGS()

    if not args.test:

        generator = tf.random.Generator.from_seed(args.seed)
        def train_augment_tf_image(image: tf.Tensor, label: tf.Tensor):
            if generator.uniform([]) >= 0.5:
                image = tf.image.flip_left_right(image)
            image = tf.image.resize_with_crop_or_pad(image, CAGS.H + 6, CAGS.W + 6)
            image = tf.image.resize(image, [generator.uniform([], CAGS.H, CAGS.H + 12 + 1, dtype=tf.int32),
                                            generator.uniform([], CAGS.W, CAGS.W + 12 + 1, dtype=tf.int32)])
            image = tf.image.crop_to_bounding_box(
                image, target_height=CAGS.H, target_width=CAGS.W,
                offset_height=generator.uniform([], maxval=tf.shape(image)[0] - CAGS.H + 1, dtype=tf.int32),
                offset_width=generator.uniform([], maxval=tf.shape(image)[1] - CAGS.W + 1, dtype=tf.int32),
            )
            return image, label

        train = cags.train.map(lambda x: (x["image"], x["mask"]))
        train = train.shuffle(5000, seed=args.seed)
        if args.augment:
            train = train.map(train_augment_tf_image)
        train = train.batch(args.batch_size)
        train = train.prefetch(tf.data.AUTOTUNE)

        dev = cags.dev.map(lambda x: (x["image"], x["mask"]))
        dev = dev.batch(args.batch_size)
        dev = dev.prefetch(tf.data.AUTOTUNE)

        # TODO: Create the model and train it
        model = EfficientUNetV2B0()

        model.compile(
            optimizer=tf.optimizers.experimental.AdamW(jit_compile=False),
            loss=tf.losses.BinaryCrossentropy(),
            #loss=scce_with_labelsmooth,
            #loss=loss,
            #loss=tf.losses.BinaryCrossentropy(),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy"), tf.metrics.BinaryIoU(name="IoU")],
            #metrics=[tf.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.BinaryIoU(name='iou')],
        )
            
        tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)
        def save_model(epoch, logs):
            model.save(os.path.join(args.logdir, f"ep{epoch+1}.h5"), include_optimizer=False)

        with tf.device("/GPU:0"):
            model.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[tb_callback, tf.keras.callbacks.LambdaCallback(on_epoch_end=save_model)])
    
    else:
        # load trained model
        if args.model_path == "":
            print("Specify path to trained model")
            exit()
        args.logdir = "/".join(args.model_path.split("/")[:-1])

        model = EfficientUNetV2B0()
        model.load_weights(args.model_path)


    tst = cags.test.map(lambda x: x["image"])
    tst = tst.batch(args.batch_size)

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cags_segmentation.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the masks on the test set
        # ACHTUNG: following procedure requires one-channel predictions !
        #test_masks = model.predict(tst)[:, :, :, 1]
        test_masks = model.predict(tst)
        print(test_masks.shape)
        np.save(os.path.join(args.logdir, "test_masks"), (test_masks >= 0.5).astype(np.uint8))

        for mask in test_masks:
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
