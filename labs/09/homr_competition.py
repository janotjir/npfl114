#!/usr/bin/env python3
import argparse
import datetime
import os
import re
import shutil
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from homr_dataset import HOMRDataset

# Team members:
# 4c2c10df-00be-4008-8e01-1526b9225726
# dc535248-fa6c-4987-b49f-25b6ede7c87d

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

parser.add_argument("--test", default=False, action="store_true", help="Load model and only annotate test data")
parser.add_argument("--model", default="", type=str, help="Model path")

parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--beam_width", default=20, type=int, help="Beam width of the beam search decoder")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate")


class Model(tf.keras.Model):
    def __init__(self, args: argparse.Namespace) -> None:
        inputs = tf.keras.layers.Input(shape=[120, None, 1], dtype=tf.float32, ragged=True)

        # TODO: Add CNN feature extraction and adjust RNN structure

        hidden = inputs.to_tensor()
        hidden = tf.transpose(hidden, [0, 2, 1, 3])
        print(hidden.shape)

        hidden = tf.keras.layers.Conv2D(8, 5, padding='same')(hidden)
        hidden = tf.keras.layers.Conv2D(8, 3, padding='same')(hidden)
        hidden = tf.keras.layers.Conv2D(8, 3, (2, 2), padding='same')(hidden)

        hidden = tf.keras.layers.Conv2D(16, 3, padding='same')(hidden)
        hidden = tf.keras.layers.Conv2D(16, 3, padding='same')(hidden)
        hidden = tf.keras.layers.Conv2D(16, 3, (4, 4), padding='same')(hidden)

        hidden = tf.keras.layers.Conv2D(32, 3, padding='same')(hidden)
        hidden = tf.keras.layers.Conv2D(32, 3, padding='same')(hidden)
        hidden = tf.keras.layers.Conv2D(32, 3, (4, 4), padding='same')(hidden)

        #hidden = tf.RaggedTensor.from_tensor(hidden, inputs.row_lengths(1))
        print(hidden.shape)
        #hidden = tf.concat(tf.split(hidden, tf.shape(hidden)[2], 2), 1)
        hidden = tf.reshape(hidden, [tf.shape(hidden)[0], tf.shape(hidden)[1], tf.shape(hidden)[2] * tf.shape(hidden)[3]])
        print(hidden.shape)

        layer = tf.keras.layers.LSTM(32, return_sequences=True)
        hidden = tf.keras.layers.Bidirectional(layer, "sum")(hidden)

        #layer = tf.keras.layers.LSTM(1024, return_sequences=True)
        #hidden += tf.keras.layers.Bidirectional(layer, "sum")(hidden)

        layer = tf.keras.layers.LSTM(32, return_sequences=True)
        hidden = tf.keras.layers.Bidirectional(layer, "sum")(hidden)
        hidden = tf.RaggedTensor.from_tensor(hidden, inputs.row_lengths())

        logits = tf.keras.layers.Dense(len(HOMRDataset.MARKS)+1, activation=None)(hidden)

        super().__init__(inputs=inputs, outputs=logits)

        # We compile the model with the CTC loss and EditDistance metric.
        self.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate, jit_compile=False),
                     loss=self.ctc_loss,
                     metrics=[HOMRDataset.EditDistanceMetric()])

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)
        self.beam_width = args.beam_width

    def ctc_loss(self, gold_labels: tf.RaggedTensor, logits: tf.RaggedTensor) -> tf.Tensor:
        assert isinstance(gold_labels, tf.RaggedTensor), "Gold labels given to CTC loss must be RaggedTensors"
        assert isinstance(logits, tf.RaggedTensor), "Logits given to CTC loss must be RaggedTensors"

        # Use tf.nn.ctc_loss to compute the CTC loss.
        # - Convert the `gold_labels` to SparseTensor and pass `None` as `label_length`.
        # - Convert `logits` to a dense Tensor and then either transpose the
        #   logits to `[max_audio_length, batch, dim]` or set `logits_time_major=False`
        # - Use `logits.row_lengths()` method to obtain the `logit_length`
        # - Use the last class (the one with the highest index) as the `blank_index`.
        #
        # The `tf.nn.ctc_loss` returns a value for a single batch example, so average
        # them to produce a single value and return it.
        loss = tf.nn.ctc_loss(
            tf.cast(gold_labels, tf.int32).to_sparse(),
            logits.to_tensor(),
            None,
            tf.cast(logits.row_lengths(), tf.int32),
            False,
            blank_index=len(HOMRDataset.MARKS)
        )
        return tf.reduce_mean(loss)

    def ctc_decode(self, logits: tf.RaggedTensor) -> tf.RaggedTensor:
        assert isinstance(logits, tf.RaggedTensor), "Logits given to CTC predict must be RaggedTensors"

        # Run `tf.nn.ctc_greedy_decoder` or `tf.nn.ctc_beam_search_decoder`
        # to perform prediction.
        # - Convert the `logits` to a dense Tensor and then transpose them
        #   to shape `[max_audio_length, batch, dim]` using `tf.transpose`
        # - Use `logits.row_lengths()` method to obtain the `sequence_length`
        # - Convert the result of the decoded from a SparseTensor to a RaggedTensor
        predictions, log_probs = tf.nn.ctc_beam_search_decoder(
            tf.transpose(logits.to_tensor(), [1, 0, 2]),
            tf.cast(logits.row_lengths(), tf.int32),
            beam_width=self.beam_width
        )
        predictions = tf.RaggedTensor.from_sparse(predictions[0])

        assert isinstance(predictions, tf.RaggedTensor), "CTC predictions must be RaggedTensors"
        return predictions

    # We override the `train_step` method, because we do not want to
    # evaluate the training data for performance reasons
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return {"loss": metric.result() for metric in self.metrics if metric.name == "loss"}

    # We override `predict_step` to run CTC decoding during prediction.
    def predict_step(self, data):
        data = data[0] if isinstance(data, tuple) else data
        y_pred = self(data, training=False)
        y_pred = self.ctc_decode(y_pred)
        return y_pred

    # We override `test_step` to run CTC decoding during evaluation.
    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        self.compute_loss(x, y, y_pred)
        y_pred = self.ctc_decode(y_pred)
        return self.compute_metrics(x, y, y_pred, None)


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.debug:
        tf.config.run_functions_eagerly(True)
        # tf.data.experimental.enable_debug_mode()

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    ))

    # Load the data. The "image" is a grayscale image represented using
    # a single channel of `tf.uint8`s in [0-255] range.
    homr = HOMRDataset()

    #for example in homr.dev:
    #    print(example['image'].shape)

    def create_dataset(name):
        def prepare_example(example):
            # Create suitable batch examples.
            # - example["mfccs"] should be used as input
            # - the example["sentence"] is a UTF-8-encoded string with the target sentence
            #   - split it to unicode characters by using `tf.strings.unicode_split`
            #   - then pass it through the `cvcs.letters_mapping` layer to map
            #     the unicode characters to ids
            img = tf.image.convert_image_dtype(example["image"], tf.float32)
            img = tf.image.resize(img, [120, 3000], preserve_aspect_ratio=True)
            ids = example["marks"]
            return img, ids

        dataset = getattr(homr, name).map(prepare_example)
        dataset = dataset.shuffle(len(dataset), seed=args.seed) if name == "train" else dataset
        dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    train, dev, test = create_dataset("train"), create_dataset("dev"), create_dataset("test")

    # TODO: Create the model and train it
    os.makedirs(args.logdir, exist_ok=True)
    shutil.copy(__file__, os.path.join(args.logdir, "code_snapshot.py")) 

    model = Model(args)
    if not args.test:
        if args.model != "":
            model.load_weights(args.model)
        
        def save_model(epoch, logs):
            model.save(os.path.join(args.logdir, f"ep{epoch+1}.h5"), include_optimizer=False)

        logs = model.fit(train, epochs=args.epochs, validation_data=dev,
                            callbacks=[model.tb_callback, tf.keras.callbacks.LambdaCallback(on_epoch_end=save_model)])
    
    else:
        model.load_weights(args.model)
        args.logdir = "/".join(args.model.split("/")[:-1])

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    with open(os.path.join(args.logdir, "homr_competition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the sequences of recognized marks.
        predictions = model.predict(test)

        for sequence in predictions:
            print(" ".join(homr.MARKS[mark] for mark in sequence), file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
