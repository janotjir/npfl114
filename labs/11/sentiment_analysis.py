#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf
import transformers

from text_classification_dataset import TextClassificationDataset


# Team members:
# 4c2c10df-00be-4008-8e01-1526b9225726
# dc535248-fa6c-4987-b49f-25b6ede7c87d


# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

parser.add_argument("--test", default=False, action="store_true", help="Load model and only annotate test data")
parser.add_argument("--model", default="", type=str, help="Model path")


class SentimentModel(tf.keras.Model):
    def __init__(self, args, train, eleczech) -> None:

        inputs = (tf.keras.layers.Input(shape=[None], dtype=tf.int32, ragged=True), tf.keras.layers.Input(shape=[None], dtype=tf.int32, ragged=True))

        # get output of eleczech's last layer
        eleczech_output = eleczech(input_ids=inputs[0].to_tensor(), attention_mask=inputs[1].to_tensor())
        
        # output is in shape [bs, max_len, 256]
        #TODO what token to choose for classification (the first one = CLS probably, according to slide 41)
        eleczech_output = eleczech_output.last_hidden_state[:, 0]
 
        # eleczech_output = tf.keras.layers.Dropout(0.2)(eleczech_output)
        
        predictions = tf.keras.layers.Dense(3, activation=tf.nn.softmax)(eleczech_output)

        super().__init__(inputs=inputs, outputs=predictions)

        self.compile(
            optimizer=tf.optimizers.experimental.Adam(learning_rate=1e-4, jit_compile=False),
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")]
            )


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

    # Load the Electra Czech small lowercased
    tokenizer = transformers.AutoTokenizer.from_pretrained("ufal/eleczech-lc-small")
    eleczech = transformers.TFAutoModel.from_pretrained("ufal/eleczech-lc-small")

    # TODO: Load the data. Consider providing a `tokenizer` to the
    # constructor of the `TextClassificationDataset`.
    facebook = TextClassificationDataset("czech_facebook", tokenizer.encode)

    def extract_data(example):
        tokens = example["tokens"]
        attention_mask = tf.ones(tf.shape(tokens), tf.int32)
        labels = facebook.train.label_mapping(example["labels"])
        return (tokens, attention_mask), labels

    def extract_data_tst(example):
        tokens = example["tokens"]
        attention_mask = tf.ones(tf.shape(tokens), tf.int32)
        return (tokens, attention_mask), None

    def create_dataset(name):
        dataset = getattr(facebook, name).dataset
        if name == "test":
            dataset = dataset.map(extract_data_tst)
        else:
            dataset = dataset.map(extract_data)
        dataset = dataset.shuffle(len(dataset), seed=args.seed) if name == "train" else dataset
        dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    train, dev = create_dataset("train"), create_dataset("dev")

    if not args.test:

        # TODO: Create the model and train it
        model = SentimentModel(args, facebook.train, eleczech)

        tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)
        def save_model(epoch, logs):
            model.save(os.path.join(args.logdir, f"ep{epoch+1}.h5"), include_optimizer=False)

        model.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[tb_callback, tf.keras.callbacks.LambdaCallback(on_epoch_end=save_model)])
    
    else:
        args.logdir = "/".join(args.model.split("/")[:-1])
        model = SentimentModel(args, facebook.train, eleczech)
        model.load_weights(args.model)

    test = create_dataset("test")
    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "sentiment_analysis.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the tags on the test set.
        predictions = model.predict(test)

        label_strings = facebook.test.label_mapping.get_vocabulary()
        for sentence in predictions:
            print(label_strings[np.argmax(sentence)], file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
