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
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--augment", default=False, action='store_true', help='Augment the training data')
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--test", default=False, action="store_true", help="Run model on test dataset")
parser.add_argument("--model_path", default="", type=str, help="Specify path to trained model")
parser.add_argument("--label_smoothing", default=0.0, type=float, help="Specify label smoothing coefficient")


# Team members:
# 4c2c10df-00be-4008-8e01-1526b9225726
# dc535248-fa6c-4987-b49f-25b6ede7c87d


def scce_with_labelsmooth(y, y_hat):
    y = tf.squeeze(tf.one_hot(tf.cast(y, tf.int32), len(CAGS.LABELS)), axis=-2)
    return tf.keras.losses.categorical_crossentropy(y, y_hat, label_smoothing=args.label_smoothing)


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

    # Load the data. Note that both the "image" and the "mask" images
    # are represented using `tf.uint8`s in [0-255] range.
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

        train = cags.train.map(lambda x: (x["image"], x["label"]))
        train = train.shuffle(5000, seed=args.seed)
        if args.augment:
            train = train.map(train_augment_tf_image)
        train = train.batch(args.batch_size)
        train = train.prefetch(tf.data.AUTOTUNE)

        dev = cags.dev.map(lambda x: (x["image"], x["label"]))
        dev = dev.batch(args.batch_size)
        dev = dev.prefetch(tf.data.AUTOTUNE)

        # Load the EfficientNetV2-B0 model
        backbone = tf.keras.applications.EfficientNetV2B0(include_top=False, pooling="avg")
        backbone.trainable = False

        # TODO: Create the model and train it
        inputs = tf.keras.layers.Input(shape=[CAGS.H, CAGS.W, CAGS.C])
        hidden = backbone(inputs)
        outputs = tf.keras.layers.Dense(len(CAGS.LABELS), activation=tf.nn.softmax)(hidden)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # TODO: Fit model
        if args.label_smoothing > 0:
            loss = scce_with_labelsmooth
        else:
            loss = tf.losses.SparseCategoricalCrossentropy()
        model.compile(
            optimizer=tf.optimizers.experimental.AdamW(jit_compile=False),
            #loss=tf.losses.SparseCategoricalCrossentropy(),
            #loss=scce_with_labelsmooth,
            loss=loss,
            metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        
        tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)
        def save_model(epoch, logs):
            model.save(os.path.join(args.logdir, f"ep{epoch+1}.h5"), include_optimizer=False)

        model.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[tb_callback, tf.keras.callbacks.LambdaCallback(on_epoch_end=save_model)])

    else:
        # load trained model
        if args.model_path == "":
            print("Specify path to trained model")
            exit()
        args.logdir = "/".join(args.model_path.split("/")[:-1])
        print(args.logdir)
        model = tf.keras.models.load_model(args.model_path)

    
    tst = cags.test.map(lambda x: x["image"])
    tst = tst.batch(args.batch_size)

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cags_classification.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the probabilities on the test set
        test_probabilities = model.predict(tst)

        for probs in test_probabilities:
            print(np.argmax(probs), file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
