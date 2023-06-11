#!/usr/bin/env python3
import argparse
import datetime
import os
import re
from typing import Dict
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from morpho_dataset import MorphoDataset


# Team members:
# 4c2c10df-00be-4008-8e01-1526b9225726
# dc535248-fa6c-4987-b49f-25b6ede7c87d


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--max_sentences", default=None, type=int, help="Maximum number of sentences to load.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--transformer_dropout", default=0., type=float, help="Transformer dropout.")
parser.add_argument("--transformer_expansion", default=4, type=float, help="Transformer FFN expansion factor.")
parser.add_argument("--transformer_heads", default=4, type=int, help="Transformer heads.")
parser.add_argument("--transformer_layers", default=2, type=int, help="Transformer layers.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--we_dim", default=64, type=int, help="Word embedding dimension.")
# If you add more arguments, ReCodEx will keep them with your default values.


class Model(tf.keras.Model):
    class FFN(tf.keras.layers.Layer):
        def __init__(self, dim, expansion, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.dim, self.expansion = dim, expansion
            # TODO: Create the required layers -- first a ReLU-activated dense
            # layer with `dim * expansion` units, followed by a dense layer
            # with `dim` units without an activation.
            # raise NotImplementedError()
            self.expanding_dense = tf.keras.layers.Dense(dim * expansion, activation='relu')
            self.squishing_dense = tf.keras.layers.Dense(dim, activation=None)

        def call(self, inputs):
            # TODO: Execute the FFN Transformer layer.
            # raise NotImplementedError()
            x = self.expanding_dense(inputs)
            x = self.squishing_dense(x)

            return x

    class SelfAttention(tf.keras.layers.Layer):
        def __init__(self, dim, heads, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.dim, self.heads = dim, heads
            # TODO: Create weight matrices W_Q, W_K, W_V, and W_O using `self.add_weight`,
            # each with shape `[dim, dim]`; keep the default for other `add_weight` arguments
            # (which means trainable float32 matrices initialized with `"glorot_uniform"`).
            # raise NotImplementedError()
            self.W_Q = self.add_weight(name="W_Q", shape=[dim, dim])
            self.W_K = self.add_weight(name="W_K", shape=[dim, dim])
            self.W_V = self.add_weight(name="W_V", shape=[dim, dim])
            self.W_O = self.add_weight(name="W_O", shape=[dim, dim])

        def call(self, inputs, mask):
            # TODO: Execute the self-attention layer.
            #
            # Start by computing Q, K and V. In all cases:
            # - first multiply `inputs` by the corresponding weight matrix W_Q/W_K/W_V,
            # - reshape via `tf.reshape` to `[batch_size, max_sentence_len, heads, dim // heads]`,
            # - transpose via `tf.transpose` to `[batch_size, heads, max_sentence_len, dim // heads]`.
            bs, max_sentence_len = tf.shape(inputs)[0], tf.shape(inputs)[1]

            Q = inputs @ self.W_Q
            Q = tf.reshape(Q, [bs, max_sentence_len, self.heads, self.dim // self.heads])
            Q = tf.transpose(Q, perm=[0, 2, 1, 3])

            K = inputs @ self.W_K
            K = tf.reshape(K, [bs, max_sentence_len, self.heads, self.dim // self.heads])
            K = tf.transpose(K, perm=[0, 2, 1, 3])

            V = inputs @ self.W_V
            V = tf.reshape(V, [bs, max_sentence_len, self.heads, self.dim // self.heads])
            V = tf.transpose(V, perm=[0, 2, 1, 3])

            # TODO: Continue by computing the self-attention weights as Q @ K^T,
            # normalizing by the square root of `dim // heads`.
            attention_weights = Q @ tf.transpose(K, [0, 1, 3, 2]) / tf.math.sqrt(self.dim / self.heads)

            # TODO: Apply the softmax, but including a suitable mask ignoring all padding words.
            # The original `mask` is a bool matrix of shape `[batch_size, max_sentence_len]`
            # indicating which words are valid (`True`) or padding (`False`).
            # To mask an input to softmax, replace it by -1e9 (theoretically we should use
            # minus infinity, but `tf.math.exp(-1e9)` is also zero because of limited precision).
            # attention_weights = tf.where(mask, attention_weights, -1e9)
            # attention_weights = tf.nn.softmax(attention_weights)
            attention_weights = tf.keras.layers.Softmax()(attention_weights, mask=mask[:, tf.newaxis, tf.newaxis, :])

            # TODO: Finally,
            # - take a weighted combination of values V according to the computed attention
            #   (using a suitable matrix multiplication),
            # - transpose the result to `[batch_size, max_sentence_len, heads, dim // heads]`,
            # - reshape to `[batch_size, max_sentence_len, dim]`,
            # - multiply the result by the W_O matrix.
            # raise NotImplementedError()
            attention_weights = attention_weights @ V
            attention_weights = tf.transpose(attention_weights, [0, 2, 1, 3])
            attention_weights = tf.reshape(attention_weights, [bs, max_sentence_len, self.dim])
            
            return attention_weights @ self.W_O

    class PositionalEmbedding(tf.keras.layers.Layer):
        def __init__(self, dim, *args, **kwargs):
            assert dim % 2 == 0  # The `dim` needs to be even to have the same number of sin&cos.
            super().__init__(*args, **kwargs)
            self.dim = dim

        def call(self, inputs):
            # TODO: Compute the sinusoidal positional embeddings. Recalling that `self.dim` is even,
            # the embeddings have a shape `[max_sentence_len, self.dim]`, and for `0 <= i < dim/2`:
            # - the value on index `[pos, i]` should be
            #     `sin(pos / 10_000 ** (2 * i / self.dim))`
            # - the value on index `[pos, self.dim/2 + i]` should be
            #     `cos(pos / 10_000 ** (2 * i / self.dim))`
            # - the `0 <= pos < max_sentence_len` is the sentence index.
            # This order is the same as in the visualization on the slides, but
            # different from the original paper where `sin` and `cos` interleave.
            # raise NotImplementedError()
            bs = int(tf.shape(inputs)[0])
            max_sentence_len = int(tf.shape(inputs)[1])
            pos_embeddings = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

            for pos in range(max_sentence_len):
                sins = tf.math.sin(tf.cast(pos, tf.float32) / 10000 ** (2 * tf.range(0, self.dim/2) / self.dim))
                sins = tf.expand_dims(sins, axis=0)
                coss = tf.math.cos(tf.cast(pos, tf.float32) / 10000 ** (2 * tf.range(0, self.dim/2) / self.dim))
                coss = tf.expand_dims(coss, axis=0)
                pos_embeddings = pos_embeddings.write(pos, tf.concat([sins, coss], axis=1))

            #tf.print(pos_embeddings.stack(), tf.shape(pos_embeddings.stack()))

            return tf.transpose(pos_embeddings.stack(), perm=[1, 0, 2])

    class Transformer(tf.keras.layers.Layer):
        def __init__(self, layers, dim, expansion, heads, dropout, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.layers, self.dim, self.expansion, self.heads, self.dropout = layers, dim, expansion, heads, dropout
            # TODO: Create:
            # - the positional embedding layer;
            # - the required number of transformer layers, each consisting of
            #   - a layer normalization and a self-attention layer followed by a dropout layer,
            #   - a layer normalization and a FFN layer followed by a dropout layer.
            self.PE_layer = Model.PositionalEmbedding(dim)
            self.SA_layers = []
            self.FFN_layers = []

            for i in range(layers):
                self.SA_layers.append([tf.keras.layers.LayerNormalization(), Model.SelfAttention(dim, heads), tf.keras.layers.Dropout(dropout)])
                self.FFN_layers.append([tf.keras.layers.LayerNormalization(), Model.FFN(dim, expansion), tf.keras.layers.Dropout(dropout)])

        def call(self, inputs, mask):
            # TODO: First compute the positional embeddings.
            pos_embeddings = self.PE_layer(inputs)
            x = inputs + pos_embeddings

            # TODO: Add the positional embeddings to the `inputs` and then
            # perform the given number of transformer layers, composed of
            # - a self-attention sub-layer, followed by
            # - a FFN sub-layer.
            # In each sub-layer, pass the input through LayerNorm, then compute
            # the corresponding operation, apply dropout, and finally add this result
            # to the original sub-layer input. Note that the given `mask` should be
            # passed to the self-attention operation to ignore the padding words.
            # raise NotImplementedError()
            for i in range(self.layers):
                res = tf.identity(x)
                #x = tf.keras.layers.LayerNormalization()(x)
                x = self.SA_layers[i][0](x)
                x = self.SA_layers[i][1](x, mask)
                #x = tf.keras.layers.Dropout(self.dropout)(x)
                x = self.SA_layers[i][2](x)
                x += res

                res = tf.identity(x)
                # x = tf.keras.layers.LayerNormalization()(x)
                x = self.FFN_layers[i][0](x)
                x = self.FFN_layers[i][1](x)
                # x = tf.keras.layers.Dropout(self.dropout)(x)
                x = self.FFN_layers[i][2](x)
                x += res

            return x

    def __init__(self, args, train):
        # Implement a transformer encoder network. The input `words` is
        # a RaggedTensor of strings, each batch example being a list of words.
        words = tf.keras.layers.Input(shape=[None], dtype=tf.string, ragged=True)

        # TODO(tagger_we): Map strings in `words` to indices by using the `word_mapping` of `train.forms`.
        indices = train.forms.word_mapping(words)

        # TODO(tagger_we): Embed input words with dimensionality `args.we_dim`. Note that the `word_mapping`
        # provides a `vocabulary_size()` call returning the number of unique words in the mapping.
        embeddings = tf.keras.layers.Embedding(train.forms.word_mapping.vocabulary_size(), args.we_dim)(indices)

        # TODO: Call the Transformer layer:
        # - create a `Model.Transformer` layer, using suitable options from `args`
        #   (using `args.we_dim` for the `dim` argument),
        # - when calling the layer, convert the ragged tensor with the input words embedding
        #   to a dense one, and also pass the following argument as a mask:
        #     `mask=tf.sequence_mask(ragged_tensor_with_input_words_embeddings.row_lengths())`
        # - finally, convert the result back to a ragged tensor.
        mask = tf.sequence_mask(embeddings.row_lengths())
        x = Model.Transformer(args.transformer_layers, args.we_dim, args.transformer_expansion, args.transformer_heads, args.transformer_dropout)(embeddings.to_tensor(), mask)
        x = tf.RaggedTensor.from_tensor(x, embeddings.row_lengths())

        # TODO(tagger_we): Add a softmax classification layer into as many classes as there are unique
        # tags in the `word_mapping` of `train.tags`. Note that the Dense layer can process
        # a `RaggedTensor` without any problem.
        predictions = tf.keras.layers.Dense(train.tags.word_mapping.vocabulary_size(), activation=tf.nn.softmax)(x)

        # Check that the created predictions are a 3D tensor.
        assert predictions.shape.rank == 3

        super().__init__(inputs=words, outputs=predictions)

        def ragged_sparse_categorical_crossentropy(y_true, y_pred):
            return tf.losses.SparseCategoricalCrossentropy()(y_true.values, y_pred.values)

        self.compile(optimizer=tf.optimizers.Adam(jit_compile=False),
                     loss=ragged_sparse_categorical_crossentropy,
                     metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")])

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)


def main(args: argparse.Namespace) -> Dict[str, float]:
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
    morpho = MorphoDataset("czech_cac", max_sentences=args.max_sentences)

    # Create the model and train
    model = Model(args, morpho.train)

    # TODO(tagger_we): Construct the data for the model, each consisting of the following pair:
    # - a tensor of string words (forms) as input,
    # - a tensor of integer tag ids as targets.
    # To create the tag ids, use the `word_mapping` of `morpho.train.tags`.
    def extract_tagging_data(example):
        # raise NotImplementedError()
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

    logs = model.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[model.tb_callback])

    # Return development and training losses for ReCodEx to validate.
    return {metric: values[-1] for metric, values in logs.history.items() if "loss" in metric}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
