#!/usr/bin/env python3
import argparse
import datetime
import os
import re

# Team members:
# 4c2c10df-00be-4008-8e01-1526b9225726
# dc535248-fa6c-4987-b49f-25b6ede7c87d

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

from morpho_analyzer import MorphoAnalyzer
from morpho_dataset import MorphoDataset

# Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use")

parser.add_argument("--test", default=False, action="store_true", help="Load model and only annotate test data")
parser.add_argument("--eval", default=False, action="store_true", help="Load model and only annotate dev data")
parser.add_argument("--model", default="", type=str, help="Model path")

# TODO: Subject to tuning
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size")
parser.add_argument("--tuner", default="Hyperband", choices=["Hyperband", "Bayes"], help="RNN layer type.")
parser.add_argument("--rnn", default="LSTM", choices=["LSTM", "GRU"], help="RNN layer type.")

morpho = MorphoDataset("czech_pdt")

class Model(tf.keras.Model):
    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset) -> None:
        # Implement a one-layer RNN network. The input `words` is
        # a `RaggedTensor` of strings, each batch example being a list of words.
        words = tf.keras.layers.Input(shape=[None], dtype=tf.string, ragged=True)

        # (tagger_we): Map strings in `words` to indices by using the `word_mapping` of `train.forms`.
        indices = train.forms.word_mapping(words)

        # With a probability of `args.word_masking`, replace the input word by an
        # unknown word (which has index 0).
        scores = tf.ones_like(indices, dtype=tf.float32)
        scores = tf.keras.layers.Dropout(args.word_masking)(scores)
        scores = tf.cast(scores, tf.int64)
        indices = indices * scores

        # (tagger_we): Embed input words with dimensionality `args.we_dim`. Note that the `word_mapping`
        # provides a `vocabulary_size()` call returning the number of unique words in the mapping.
        w_embeddings = tf.keras.layers.Embedding(train.forms.word_mapping.vocabulary_size(), args.we_dim)(indices)

        # Create a vector of input words from all batches using `words.values`
        # and pass it through `tf.unique`, obtaining a list of unique words and
        # indices of the original flattened words in the unique word list.
        unq, idx = tf.unique(words.values)

        # Create sequences of letters by passing the unique words through
        # `tf.strings.unicode_split` call; use "UTF-8" as `input_encoding`.
        letter_sq = tf.strings.unicode_split(unq, 'UTF-8')

        # Map the letters into ids by using `char_mapping` of `train.forms`.
        ids = train.forms.char_mapping(letter_sq)

        # Embed the input characters with dimensionality `args.cle_dim`.
        c_embeddings = tf.keras.layers.Embedding(train.forms.char_mapping.vocabulary_size(), args.cle_dim)(ids)

        # Pass the embedded letters through a bidirectional GRU layer
        # with dimensionality `args.cle_dim`, obtaining character-level representations
        # of the whole words, **concatenating** the outputs of the forward and backward RNNs.
        gru = tf.keras.layers.GRU(args.cle_dim, return_sequences=False)
        c_embeddings = tf.keras.layers.Bidirectional(gru, "concat")(c_embeddings.to_tensor())

        # Use `tf.gather` with the indices generated by `tf.unique` to transform
        # the computed character-level representations of the unique words to representations
        # of the flattened (non-unique) words.
        c_embeddings = tf.gather(c_embeddings, idx)

        # Then, convert these character-level word representations into
        # a RaggedTensor of the same shape as `words` using `words.with_values` call.
        c_embeddings = words.with_values(c_embeddings)

        # Concatenate the word-level embeddings and the computed character-level WEs
        # (in this order).
        embeddings = tf.keras.layers.concatenate([w_embeddings, c_embeddings], axis=-1)

        # (tagger_we): Create the specified `args.rnn` RNN layer (LSTM, GRU) with
        # dimension `args.rnn_dim`. The layer should produce an output for every
        # sequence element (so a 3D output). Then apply it in a bidirectional way on
        # the word representations, **summing** the outputs of forward and backward RNNs.
        layer = getattr(tf.keras.layers, args.rnn)(args.rnn_dim, return_sequences=True)
        hidden = tf.keras.layers.Bidirectional(layer, "sum")(embeddings.to_tensor())
        hidden = tf.RaggedTensor.from_tensor(hidden, embeddings.row_lengths())

        # (tagger_we): Add a softmax classification layer into as many classes as there are unique
        # tags in the `word_mapping` of `train.tags`. Note that the Dense layer can process
        # a `RaggedTensor` without any problem.
        predictions = tf.keras.layers.Dense(train.tags.word_mapping.vocabulary_size(), activation=tf.nn.softmax)(hidden)

        # Check that the created predictions are a 3D tensor.
        assert predictions.shape.rank == 3

        super().__init__(inputs=words, outputs=predictions)

        def ragged_sparse_categorical_crossentropy(y_true, y_pred):
            return tf.losses.SparseCategoricalCrossentropy()(y_true.values, y_pred.values)

        self.compile(optimizer=tf.optimizers.Adam(jit_compile=False),
                     loss=ragged_sparse_categorical_crossentropy,
                     metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")])

        #self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)


def model_builder(hp):
        args.cle_dim = hp.Choice('cle_dim', values=[16, 32, 64])
        args.we_dim = hp.Choice('we_dim', values=[16, 32, 64])
        args.rnn_dim = hp.Choice('rnn_dim', values=[32, 64, 128])
        args.word_masking = hp.Choice('word_masking', values=[0.1, 0.2, 0.5])
        if args.tuner == "Hyperband":
            args.rnn = hp.Choice('rnn', values=["LSTM", "GRU"])

        model = Model(args, morpho.train)

        return model


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    tf.keras.utils.set_random_seed(args.seed)
    #tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    #tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.debug:
        tf.config.run_functions_eagerly(True)
        tf.data.experimental.enable_debug_mode()

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the data. Using analyses is only optional.
    analyses = MorphoAnalyzer("czech_pdt_analyses")

    # TODO: Consider utilising analysis outputs

    def extract_tagging_data(example):
        forms = example['forms']
        tags = example['tags']
        ids = morpho.train.tags.word_mapping(tags)
        return forms, ids

    def create_dataset(name):
        dataset = getattr(morpho, name).dataset
        dataset = dataset.map(extract_tagging_data)
        dataset = dataset.shuffle(len(dataset), seed=args.seed) if name == "train" else dataset
        dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    train, dev = create_dataset("train"), create_dataset("dev")

    # Training
    if not args.test and not args.eval:
        if args.tuner == "Hyperband":
            tuner = kt.Hyperband(model_builder, 'val_accuracy', max_epochs=args.epochs, project_name="hyperband")
        else:
            tuner = kt.BayesianOptimization(model_builder, 'val_accuracy', max_trials=50, project_name="bayes")

        tuner.search(train, epochs=args.epochs, validation_data=dev)

        best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
        print(best_hps)
        model = tuner.hypermodel.build(best_hps)
        history = model.fit(train, epochs=30, validation_data=dev)

        val_acc_per_epoch = history.history['val_accuracy']
        best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
        print('Best epoch: %d' % (best_epoch,))

        best_model = tuner.hypermodel.build(best_hps)
        best_model.fit(train, epochs=best_epoch, validation_data=dev)

        best_model.save('the_best_fkin_model.h5', include_optimizer=False)
        eval_result = best_model.evaluate(dev)
        print(eval_result)
        
    else:
        model = Model(args, morpho.train)
        model.load_weights(args.model)
        args.logdir = "/".join(args.model.split("/")[:-1])
    if args.eval:
        test = create_dataset("dev")
        filename = "tagger_competition_val.txt"
    else:
        test = create_dataset("test")
        filename = "tagger_competition.txt"

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, filename), "w", encoding="utf-8") as predictions_file:
        # Predict the tags on the test set; update the following code
        # if you use other output structure than in tagger_we.
        predictions = model.predict(test)

        tag_strings = morpho.train.tags.word_mapping.get_vocabulary()
        for sentence in predictions:
            for word in np.asarray(sentence):
                print(tag_strings[np.argmax(word)], file=predictions_file)
            print(file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
