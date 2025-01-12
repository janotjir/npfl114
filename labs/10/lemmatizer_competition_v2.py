#!/usr/bin/env python3
import argparse
import datetime
import os
import re
import shutil
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf
tf.get_logger().addFilter(lambda m: "Analyzer.lamba_check" not in m.getMessage())  # Avoid pesky warning

from morpho_analyzer import MorphoAnalyzer
from morpho_dataset import MorphoDataset

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--rnn_dim", default=256, type=int, help="RNN layer dimension.")
parser.add_argument("--emb_dim", default=256, type=int, help="Embedding dimension.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

parser.add_argument("--test", default=False, action="store_true", help="Load model and only annotate test data")
parser.add_argument("--eval", default=False, action="store_true", help="Load model and only annotate dev data")
parser.add_argument("--model", default="", type=str, help="Model path")

# Team members:
# 4c2c10df-00be-4008-8e01-1526b9225726
# dc535248-fa6c-4987-b49f-25b6ede7c87d


class WithAttention(tf.keras.layers.AbstractRNNCell):
    """A class adding Bahdanau attention to the given RNN cell."""
    def __init__(self, cell, attention_dim):
        super().__init__()
        self._cell = cell

        self._project_encoder_layer = tf.keras.layers.Dense(attention_dim)
        self._project_decoder_layer = tf.keras.layers.Dense(attention_dim)
        self._output_layer = tf.keras.layers.Dense(1)

    @property
    def state_size(self):
        return self._cell.state_size

    def setup_memory(self, encoded):
        self._encoded = encoded
        self._encoded_projected = self._project_encoder_layer(encoded)

    def call(self, inputs, states):
        decoder_projected = self._project_decoder_layer(states[0])
        projected_sum = tf.expand_dims(decoder_projected, 1) + self._encoded_projected
        sum_tanh = tf.nn.tanh(projected_sum)
        attn_logits = self._output_layer(sum_tanh)
        weights = tf.nn.softmax(attn_logits, 1)
        attention = tf.reduce_sum(self._encoded * weights, 1)

        attended = tf.concat([inputs, attention], 1)
        out = self._cell(attended, states)
        return out


class Model(tf.keras.Model):
    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset) -> None:
        super().__init__()

        self._source_mapping = train.forms.char_mapping
        self._target_mapping = train.lemmas.char_mapping
        self._target_mapping_inverse = type(self._target_mapping)(vocabulary=self._target_mapping.get_vocabulary(), invert=True)

        # sum stuff for tagger
        self._word_mapping = train.forms.word_mapping
        self.tag_mapping = train.tags.word_mapping
        self.w_embeddings = tf.keras.layers.Embedding(train.forms.word_mapping.vocabulary_size(), 128)
        self.c_embed_gru1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True), 'concat')
        self.c_embed_gru2 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=False), 'concat')

        self.tag_hidden1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=True), "sum")
        self.tag_hidden2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=True), "sum")
        self.tag_hidden3 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=True), "sum")
        self.tag_final = tf.keras.layers.Dense(train.tags.word_mapping.vocabulary_size(), activation=tf.nn.softmax)
        self.tag_to_decoder = tf.keras.layers.Dense(args.rnn_dim, activation=None)

        # TODO(lemmatizer_noattn): Define
        # - `self._source_embedding` as an embedding layer of source ids into `args.cle_dim` dimensions
        embedding_dim = args.emb_dim
        self._source_embedding = tf.keras.layers.Embedding(train.forms.char_mapping.vocabulary_size(), embedding_dim)

        # TODO: Define
        # - `self._source_rnn` as a bidirectional GRU with `args.rnn_dim` units, returning **whole sequences**,
        #   summing opposite directions
        self._source_rnn = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True), merge_mode="sum")

        # TODO: Then define
        # - `self._target_rnn` as a `tf.keras.layers.RNN` returning whole sequences, utilizing the
        #   attention-enhanced cell using `WithAttention` with `attention_dim` of `args.rnn_dim`,
        #   employing the `tf.keras.layers.GRUCell` with `args.rnn_dim` units as the underlying cell.
        rnn_dim = args.rnn_dim
        self._target_rnn = tf.keras.layers.RNN(WithAttention(tf.keras.layers.GRUCell(rnn_dim), rnn_dim), return_sequences=True)

        # TODO(lemmatizer_noattn): Then define
        # - `self._target_output_layer` as a Dense layer into as many outputs as there are unique target chars
        self._target_output_layer = tf.keras.layers.Dense(train.lemmas.char_mapping.vocabulary_size())


        # TODO: tie embeddings
        self._target_output_layer.build(rnn_dim)
        # TODO(lemmatizer_noattn): Create a function `self._target_embedding` which computes the embedding of given
        # target ids. When called, use `tf.gather` to index the transposition of the shared embedding
        # matrix `self._target_output_layer.kernel` multiplied by the square root of `args.rnn_dim`.
        def tied_embedding(ids):
            embeddings = tf.gather(tf.transpose(self._target_output_layer.kernel, [1, 0]) * tf.math.sqrt(tf.cast(rnn_dim, tf.float32)), ids)
            return embeddings
        self._target_embedding = tied_embedding

        def ragged_sparse_categorical_crossentropy(y_true, y_pred):
            return tf.losses.SparseCategoricalCrossentropy()(y_true.values, y_pred.values)

        # Compile the model
        self.compile(
            optimizer=tf.optimizers.Adam(learning_rate=1e-3, jit_compile=False),
            loss={"lemmas" : tf.losses.SparseCategoricalCrossentropy(from_logits=True), "tags" : ragged_sparse_categorical_crossentropy},
            metrics=[{"lemmas" : tf.metrics.Accuracy(name="accuracy"), "tags" : None}], #tf.metrics.SparseCategoricalAccuracy(name="tag_accuracy")}],
        )

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)

    def encoder(self, inputs: tf.Tensor) -> tf.Tensor:
        # TODO(lemmatizer_noattn): Embed the inputs using `self._source_embedding`.
        embeddings = self._source_embedding(inputs)

        # TODO: Run the `self._source_rnn` on the embedded sequences, then convert its result
        # to a dense tensor using the `.to_tensor()` call, and return it.
        return self._source_rnn(embeddings).to_tensor()

    def tagger(self, inputs):
        # (tagger_we): Map strings in `words` to indices by using the `word_mapping` of `train.forms`.
        indices = self._word_mapping(inputs)

        # With a probability of `args.word_masking`, replace the input word by an
        # unknown word (which has index 0).
        scores = tf.ones_like(indices, dtype=tf.float32)
        scores = tf.keras.layers.Dropout(0.3)(scores)
        scores = tf.cast(scores, tf.int64)
        indices = indices * scores

        # (tagger_we): Embed input words with dimensionality `args.we_dim`. Note that the `word_mapping`
        # provides a `vocabulary_size()` call returning the number of unique words in the mapping.
        w_embeddings = self.w_embeddings(indices)

        # Create a vector of input words from all batches using `words.values`
        # and pass it through `tf.unique`, obtaining a list of unique words and
        # indices of the original flattened words in the unique word list.
        unq, idx = tf.unique(inputs.values)

        # Create sequences of letters by passing the unique words through
        # `tf.strings.unicode_split` call; use "UTF-8" as `input_encoding`.
        letter_sq = tf.strings.unicode_split(unq, 'UTF-8')

        # Map the letters into ids by using `char_mapping` of `train.forms`.
        ids = self._source_mapping(letter_sq)

        # Embed the input characters with dimensionality `args.cle_dim`.
        c_embeddings = self._source_embedding(ids)
        c_embeddings_ = self.c_embed_gru1(c_embeddings.to_tensor())
        c_embeddings = tf.RaggedTensor.from_tensor(c_embeddings_, c_embeddings.row_lengths())
        c_embeddings = self.c_embed_gru2(c_embeddings.to_tensor())

        # Use `tf.gather` with the indices generated by `tf.unique` to transform
        # the computed character-level representations of the unique words to representations
        # of the flattened (non-unique) words.
        c_embeddings = tf.gather(c_embeddings, idx)

        # Then, convert these character-level word representations into
        # a RaggedTensor of the same shape as `words` using `words.with_values` call.
        c_embeddings = inputs.with_values(c_embeddings)

        # Concatenate the word-level embeddings and the computed character-level WEs
        # (in this order).
        embeddings = tf.keras.layers.concatenate([w_embeddings, c_embeddings], axis=-1)

        # (tagger_we): Create the specified `args.rnn` RNN layer (LSTM, GRU) with
        # dimension `args.rnn_dim`. The layer should produce an output for every
        # sequence element (so a 3D output). Then apply it in a bidirectional way on
        # the word representations, **summing** the outputs of forward and backward RNNs.
        hidden = self.tag_hidden1(embeddings.to_tensor())
        hidden = tf.RaggedTensor.from_tensor(hidden, embeddings.row_lengths())
        hidden_ = self.tag_hidden2(hidden.to_tensor())
        hidden = tf.RaggedTensor.from_tensor(hidden_, hidden.row_lengths())
        hidden_ = self.tag_hidden3(hidden.to_tensor())
        hidden = tf.RaggedTensor.from_tensor(hidden_, hidden.row_lengths())
        #predictions = self.tag_final(hidden)
        
        return hidden

    def decoder_training(self, encoded: tf.Tensor, targets: tf.Tensor, tags) -> tf.Tensor:
        # TODO(lemmatizer_noattn): Generate inputs for the decoder, which is obtained from `targets` by
        # - prepending `MorphoDataset.BOW` as the first element of every batch example,
        # - dropping the last element of `targets` (which is `MorphoDataset.EOW`)
        inputs = tf.concat([tf.ones([tf.shape(targets)[0], 1], dtype=tf.int64) * MorphoDataset.BOW, targets[:, :-1]], axis=1)

        # TODO: Pre-compute the projected encoder states in the attention by calling
        # the `setup_memory` of the `self._target_rnn.cell` on the `encoded` input.
        self._target_rnn.cell.setup_memory(encoded)

        # TODO: Process the generated inputs by
        # - the `self._target_embedding` layer to obtain embeddings,
        # - the `self._target_rnn` layer, passing an additional parameter `initial_state=[encoded[:, 0]]`,
        # - the `self._target_output_layer` to obtain logits,
        # and return the result.
        embeddings = self._target_embedding(inputs)
        tag_codes = tf.reshape(self.tag_to_decoder(tags), [-1, tf.shape(embeddings)[-1]])
        tag_logits = self.tag_final(tags)
        #tf.print(tf.shape(embeddings), tf.shape(tag_codes))
        embeddings = tf.concat([tag_codes[:, tf.newaxis, :], embeddings[:, 1:, :]], axis=1)
        #tf.print(tf.shape(embeddings), tf.shape(tags))
        hidden = self._target_rnn(embeddings, initial_state=[encoded[:, 0]])
        logits = self._target_output_layer(hidden)
        return {"lemmas" : logits.values, "tags" : tag_logits}

    @tf.function
    def decoder_inference(self, encoded: tf.Tensor, max_length: tf.Tensor, tags) -> tf.Tensor:
        """The decoder_inference runs a while-cycle inside a computation graph.

        To that end, it needs to be explicitly marked as @tf.function, so that the
        below `while` cycle is "embedded" in the computation graph. Alternatively,
        we might explicitly use the `tf.while_loop` operation, but a native while
        cycle is more readable.
        """
        batch_size = tf.shape(encoded)[0]
        max_length = tf.cast(max_length, tf.int32)

        # TODO(decoder_training): Pre-compute the projected encoder states in the attention by calling
        # the `setup_memory` of the `self._target_rnn.cell` on the `encoded` input.
        self._target_rnn.cell.setup_memory(encoded)

        # TODO: Define the following variables, that we will use in the cycle:
        # - `index`: a scalar tensor with dtype `tf.int32` initialized to 0,
        # - `inputs`: a batch of `MorphoDataset.BOW` symbols of type `tf.int64`,
        # - `states`: initial RNN state from the encoder, i.e., `[encoded[:, 0]]`,
        index = tf.constant(0, dtype=tf.int32)
        inputs = tf.cast(tf.fill([batch_size], MorphoDataset.BOW), tf.int64)
        states = [encoded[:, 0]]

        # We collect the results from the while-cycle into the following `tf.TensorArray`,
        # which is a dynamic collection of tensors that can be written to. We also
        # create `result_lengths` containing lengths of completely generated sequences,
        # starting with `max_length` and optionally decreasing when an EOW is generated.
        result = tf.TensorArray(tf.int64, size=max_length)
        result_lengths = tf.fill([batch_size], max_length)

        while tf.math.logical_and(index < max_length, tf.math.reduce_any(result_lengths == max_length)):
            # TODO(lemmatizer_noattn):
            # - First embed the `inputs` using the `self._target_embedding` layer.
            # - Then call `self._target_rnn.cell` using two arguments, the embedded `inputs`
            #   and the current `states`. The call returns a pair of (outputs, new states),
            #   where the new states should replace the current `states`.
            # - Pass the outputs through the `self._target_output_layer`.
            # - Finally generate the most probable prediction for every batch example.
            embeddings = self._target_embedding(inputs)
            if tf.equal(index, tf.constant(0, dtype=tf.int32)):
                tag_codes = tf.reshape(self.tag_to_decoder(tags), [-1, tf.shape(embeddings)[-1]])
                inputs = tag_codes
            outputs, states = self._target_rnn.cell(embeddings, states)
            logits = self._target_output_layer(outputs)
            predictions = tf.math.argmax(logits, axis=-1)

            # Store the predictions in the `result` on the current `index`. Then update
            # the `result_lengths` by setting it to current `index` if an EOW was generated
            # for the first time.
            result = result.write(index, predictions)
            result_lengths = tf.where(tf.math.logical_and(predictions == MorphoDataset.EOW, result_lengths > index), index, result_lengths)

            # TODO(lemmatizer_noattn): Finally,
            # - set `inputs` to the `predictions`,
            # - increment the `index` by one.
            inputs = predictions
            index += 1

        # Stack the `result` into a dense rectangular tensor, and create a ragged tensor
        # from it using the `result_lengths`.
        result = tf.RaggedTensor.from_tensor(tf.transpose(result.stack()), lengths=result_lengths)
        return result

    def train_step(self, data):
        x, y, tags = data

        tags = self.tag_mapping(tags)

        # Forget about sentence boundaries and instead consider
        # all valid form-lemma pairs as independent batch examples.
        x_flat, y_flat = x.values, y.values

        # TODO(lemmatizer_noattn): Process `x_flat` by
        # - `tf.strings.unicode_split` with encoding "UTF-8" to generate a ragged
        #   tensor with individual characters as strings,
        # - `self._source_mapping` to remap the character strings to ids.
        x_flat = tf.strings.unicode_split(x_flat, "UTF-8")
        x_flat = self._source_mapping(x_flat)

        # TODO(lemmatizer_noattn): Process `y_flat` by
        # - `tf.strings.unicode_split` with encoding "UTF-8" to generate a ragged
        #   tensor with individual characters as strings,
        # - `self._target_mapping` to remap the character strings to ids,
        # - finally, append a `MorphoDataset.EOW` to the end of every batch example.
        y_flat = tf.strings.unicode_split(y_flat, "UTF-8")
        y_flat = self._target_mapping(y_flat)
        y_flat = tf.concat([y_flat, tf.ones([tf.shape(y_flat)[0], 1], dtype=tf.int64) * MorphoDataset.EOW], axis=1)

        with tf.GradientTape() as tape:
            encoded = self.encoder(x_flat)
            pred_tags = self.tagger(x)
            y_pred = self.decoder_training(encoded, y_flat, pred_tags)
            y_flat = {"lemmas": y_flat.values, "tags": tags}
            loss = self.compute_loss(x, y_flat, y_pred)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)

        return {"loss": metric.result() for metric in self.metrics if metric.name == "loss"}

    def predict_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        # As in `train_step`, forget about sentence boundaries.
        data_flat = data.values

        # TODO(lemmatizer_noattn): As in `train_step`, pass `data_flat` through
        # - `tf.strings.unicode_split` with encoding "UTF-8" to generate a ragged
        #   tensor with individual characters as strings,
        # - `self._source_mapping` to remap the character strings to ids.
        data_flat = tf.strings.unicode_split(data_flat, "UTF-8")
        data_flat = self._source_mapping(data_flat)

        pred_tags = self.tagger(data)
        #tag_logits = self.tag_final(pred_tags)
        
        encoded = self.encoder(data_flat)
        y_pred = self.decoder_inference(encoded, data_flat.bounding_shape(axis=1) + 10, pred_tags)
        y_pred = self._target_mapping_inverse(y_pred)
        y_pred = tf.strings.reduce_join(y_pred, axis=-1)

        # Finally, convert the individual lemmas back to sentences of lemmas using
        # the original sentence boundaries.
        y_pred = data.with_values(y_pred)
        #return {"lemmas" : y_pred, "tags" : tag_logits}
        return y_pred

    def test_step(self, data):
        x, y, tags = data

        y_pred = self.predict_step(x)
        self.compiled_metrics.update_state(tf.ones_like(y, dtype=tf.int32), tf.cast(y_pred == y, tf.int32))
        return {m.name: m.result() for m in self.metrics if m.name != "loss"}


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
        os.path.basename(globals().get("__file__", "notebook"))[:-3],
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    ))

    # Load the data. Using analyses is only optional.
    morpho = MorphoDataset("czech_pdt", add_bow_eow=True)
    analyses = MorphoAnalyzer("czech_pdt_analyses")

    def create_dataset(name):
        dataset = getattr(morpho, name).dataset
        dataset = dataset.map(lambda example: (example["forms"], example["lemmas"], example["tags"]))
        dataset = dataset.shuffle(len(dataset), seed=args.seed) if name == "train" else dataset
        dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
        return dataset

    if not args.test and not args.eval:
        os.makedirs(args.logdir, exist_ok=True)
        shutil.copy(__file__, os.path.join(args.logdir, "code_snapshot.py")) 

        train, dev = create_dataset("train"), create_dataset("dev")

        # TODO: Create the model and train it
        model = Model(args, morpho.train)
        if args.model != "":
            model.load_weights(args.model)
        
        def save_model(epoch, logs):
            model.save_weights(os.path.join(args.logdir, f"ep{epoch+1}.ckpt"))

        model.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[model.tb_callback, tf.keras.callbacks.LambdaCallback(on_epoch_end=save_model)])

    else:
        # load trained model
        model = Model(args, morpho.train)
        model.load_weights(args.model)
        

    if args.eval:
        test = create_dataset("dev")
        filename = "lemmatizer_competition_val.txt"
    else:
        test = create_dataset("test")
        filename = "lemmatizer_competition.txt"

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, filename), "w", encoding="utf-8") as predictions_file:
        # Predict the tags on the test set; update the following prediction
        # command if you use other output structure than in lemmatizer_noattn.
        predictions = model.predict(test)
        for sentence in predictions:
            for word in sentence:
                print(word.numpy().decode("utf-8"), file=predictions_file)
            print(file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
