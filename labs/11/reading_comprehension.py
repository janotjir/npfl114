#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf
import transformers
import os

from reading_comprehension_dataset import ReadingComprehensionDataset


# Team members:
# 4c2c10df-00be-4008-8e01-1526b9225726
# dc535248-fa6c-4987-b49f-25b6ede7c87d


# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

parser.add_argument("--test", default=False, action="store_true", help="Load model and only annotate test data")
parser.add_argument("--model", default="", type=str, help="Model path")


class RoundACC(tf.keras.metrics.Metric):
    def __init__(self, name="Rounded match accuracy", **kwargs):
        super(RoundACC, self).__init__(name=name, **kwargs)
        self.samples = self.add_weight(name="samp", initializer='zeros')
        self.correct = self.add_weight(name="ok", initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        rounded = tf.cast(tf.round(y_pred), tf.int32)
        hit = tf.reduce_all(tf.equal(y_true, rounded))
        self.samples.assign_add(tf.constant(1, dtype=tf.float32))
        self.correct.assign_add(tf.cast(hit, tf.float32))

    def result(self):
        return self.correct / self.samples

class Comprehender(tf.keras.Model):
    def __init__(self, args, boreczech) -> None:
        inputs = (tf.keras.layers.Input(shape=[None], dtype=tf.int32, ragged=True),
                  tf.keras.layers.Input(shape=[None], dtype=tf.int32, ragged=True))

        boreczech_output = boreczech(input_ids=inputs[0].to_tensor(), attention_mask=inputs[1].to_tensor()).last_hidden_state

        pred_start = tf.keras.layers.Dense(1)(boreczech_output)
        pred_start = tf.squeeze(pred_start, axis=-1)
        pred_start = tf.keras.layers.Softmax(name="a_start")(pred_start)

        pred_end = tf.keras.layers.Dense(1)(boreczech_output)
        pred_end = tf.squeeze(pred_end, axis=-1)
        pred_end = tf.keras.layers.Softmax(name="a_end")(pred_end)

        def sparse_loss(y_true, y_pred):
            y_true = tf.squeeze(y_true)
            y_true = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])

            loss = tf.keras.backend.categorical_crossentropy(y_true, y_pred)
            loss = tf.keras.backend.mean(loss)

            return loss

        super().__init__(inputs=inputs, outputs={"a_start": pred_start, "a_end": pred_end})

        self.compile(
            optimizer=tf.optimizers.experimental.Adam(learning_rate=1e-5, jit_compile=False),
            loss={"a_start":sparse_loss, 
            "a_end":sparse_loss},
            metrics={"a_start": [tf.metrics.SparseCategoricalAccuracy()], "a_end": [tf.metrics.SparseCategoricalAccuracy(name="accuracy_end")]}
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

    # Load the pre-trained RobeCzech model
    tokenizer = transformers.AutoTokenizer.from_pretrained("ufal/robeczech-base")
    robeczech = transformers.TFAutoModel.from_pretrained("ufal/robeczech-base")

    #print(robeczech.config)

    # Load the data
    loaded_data = ReadingComprehensionDataset()

    def extract_data(example):
        context_tok = tf.constant(tokenizer.encode(example["context"], max_length=512, truncation=True))
        token_list = []
        mask_list = []
        #str_list = []
        output_list = []
        for ex in example["qas"]:
            q_tok = tf.constant(tokenizer.encode(ex["question"], max_length=512,  truncation=True))
            #q_tok = tf.pad(q_tok, tf.constant([[1, 0]]), constant_values=1)
            token_ids = tf.concat([context_tok, q_tok], axis=0)
            attention_mask = tf.ones(tf.shape(token_ids), tf.int32)
            str_out = ex["answers"][0]["text"]
            # Let's say we want to predict the index of the starting word and length of the answer via regression
            out = tf.constant([int(ex["answers"][0]["start"]), int(ex["answers"][0]["start"])+len(str_out.split())-1])

            token_list.append(token_ids)
            mask_list.append(attention_mask)
            #str_list.append(str_out)
            output_list.append(out)

        return token_list, mask_list, output_list

    def extract_tst_data(example):
        context_tok = tf.constant(tokenizer.encode(example["context"], max_length=512, truncation=True))
        token_list = []
        mask_list = []
        output_list = []
        for ex in example["qas"]:
            q_tok = tf.constant(tokenizer.encode(ex["question"], max_length=512, truncation=True))
            #q_tok = tf.pad(q_tok, tf.constant([[1, 0]]), constant_values=1)
            token_ids = tf.concat([context_tok, q_tok], axis=0)
            attention_mask = tf.ones(tf.shape(token_ids), tf.int32)

            token_list.append(token_ids)
            mask_list.append(attention_mask)
            output_list.append(tf.constant([0, 0]))

        return token_list, mask_list, output_list

    def group_inputs(tokens, masks, output):
        return (tokens, masks), {"a_start":output[0], "a_end":output[1]}

    def create_dataset(name):
        dataset = getattr(loaded_data, name).paragraphs
        examp_func = extract_data
        if name == "test":
            examp_func = extract_tst_data
        in_tokens = []
        in_masks = []
        out = []
        for ex in dataset:
            tokens, masks, outs = examp_func(ex)
            in_tokens.extend(tokens)
            in_masks.extend(masks)
            out.extend(outs)
        in_tokens = tf.ragged.stack(in_tokens)
        in_masks = tf.ragged.stack(in_masks)
        output = tf.stack(out)
        dataset = tf.data.Dataset.from_tensor_slices((in_tokens, in_masks, output))
        dataset = dataset.map(group_inputs)
        dataset = dataset.shuffle(len(dataset), seed=args.seed) if name == "train" else dataset
        dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    train, dev, test = create_dataset("train"), create_dataset("dev"), create_dataset("test")

    '''if os.path.exists("train_data"):
        train = tf.data.Dataset.load("train_data")
        dev = tf.data.Dataset.load("dev_data")
        test = tf.data.Dataset.load("test_data")
        print("Data loaded")
    else:
        train, dev, test = create_dataset("train"), create_dataset("dev"), create_dataset("test")
        train.save("train_data")
        dev.save("dev_data")
        test.save("test_data")
        print("Data created")'''

    """for ex in train:
        print(ex)
        break
    exit()"""

    # TODO: Create the model and train it
    if not args.test:

        # TODO: Create the model and train it
        model = Comprehender(args, robeczech)

        tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)

        def save_model(epoch, logs):
            model.save(os.path.join(args.logdir, f"ep{epoch + 1}.h5"), include_optimizer=False)

        model.fit(train, epochs=args.epochs, validation_data=dev,
                  callbacks=[tb_callback, tf.keras.callbacks.LambdaCallback(on_epoch_end=save_model)])

    else:
        args.logdir = "/".join(args.model.split("/")[:-1])
        model = Comprehender(args, robeczech)
        model.load_weights(args.model)

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "reading_comprehension.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the answers as strings, one per line.
        predictions = ...

        for answer in predictions:
            print(answer, file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
