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
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

parser.add_argument("--test", default=False, action="store_true", help="Load model and only annotate test data")
parser.add_argument("--eval", default=False, action="store_true", help="Load model and only annotate dev data")
parser.add_argument("--model", default="", type=str, help="Model path")


class WarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, steps_per_epoch, warmup_epochs=1, epochs=10):
    super().__init__()
    self.warmup_steps = steps_per_epoch * warmup_epochs
    
    self.warmup_decay = tf.keras.optimizers.schedules.PolynomialDecay(0., steps_per_epoch, 1e-4)
    self.decay = tf.keras.optimizers.schedules.CosineDecay(1e-4, steps_per_epoch * (epochs-warmup_epochs))

  def __call__(self, step):
    step = tf.cast(step, dtype=tf.float32)
    # arg1 = tf.math.rsqrt(step)
    # arg2 = step * (self.warmup_steps ** -1.5)
    # return 5e-2 * tf.math.minimum(arg1, arg2)
    
    # return tf.cond(step < self.warmup_steps, lambda: 1e-3 * step * (self.warmup_steps ** -1.5), lambda: 1e-4)
    
    return tf.cond(step < self.warmup_steps, lambda: self.warmup_decay(step), lambda: self.decay(step - self.warmup_steps))


class Comprehender(tf.keras.Model):
    def __init__(self, args, boreczech, steps_per_epoch=None) -> None:
        inputs = (tf.keras.layers.Input(shape=[None], dtype=tf.int32, ragged=True),
                  tf.keras.layers.Input(shape=[None], dtype=tf.int32, ragged=True))

        boreczech_output = boreczech(input_ids=inputs[0].to_tensor(), attention_mask=inputs[1].to_tensor()).last_hidden_state
        
        boreczech_output = tf.keras.layers.Dropout(0.5)(boreczech_output)

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

        if steps_per_epoch is not None:
            lr = WarmupSchedule(steps_per_epoch, warmup_epochs=1, epochs=args.epochs)
        else:
            lr = 1e-3

        self.compile(
            optimizer=tf.optimizers.experimental.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9, jit_compile=False),
            loss={"a_start": tf.keras.losses.SparseCategoricalCrossentropy(), "a_end": tf.keras.losses.SparseCategoricalCrossentropy()},
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
        context = example["context"]
        token_list = []
        mask_list = []
        output_list = []
        for ex in example["qas"]:
            question = ex["question"]
            tok = tokenizer(context, question, max_length=512, truncation="only_first")
            token_ids = tok["input_ids"]
            attention_mask = tok["attention_mask"]
            start = tok.char_to_token(ex["answers"][0]["start"])
            end = tok.char_to_token(ex["answers"][0]["start"] + len(ex["answers"][0]["text"]) - 1)
            if start is None or end is None:
                #print("Skip")
                continue
            out = tf.constant([int(start), int(end)])

            token_list.append(token_ids)
            mask_list.append(attention_mask)
            output_list.append(out)

        return token_list, mask_list, output_list

    def create_tst_dataset(dataset="test"):
        if dataset == "test":
            tst = loaded_data.test.paragraphs
        else:
            tst = loaded_data.dev.paragraphs
        input_list = []
        helper_list = []
        for cxt in tst:
            context = cxt["context"]
            for ex in cxt["qas"]:
                question = ex["question"]
                tok = tokenizer(context, question, max_length=512, truncation="only_first")
                token_ids = tf.RaggedTensor.from_row_starts(tok["input_ids"], tf.constant([0]))
                attention_mask = tf.RaggedTensor.from_row_starts(tok["attention_mask"], tf.constant([0]))
                
                input_list.append((token_ids, attention_mask))
                helper_list.append((context, tok))

        return input_list, helper_list

    def group_inputs(tokens, masks, output):
        return (tokens, masks), {"a_start":output[0], "a_end":output[1]}

    def create_dataset(name):
        dataset = getattr(loaded_data, name).paragraphs
        examp_func = extract_data
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
        #dataset = dataset.shuffle(len(dataset), seed=args.seed) if name == "train" else dataset
        #dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
        #dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    # Create the model and train it
    if not args.test and not args.eval:
        if os.path.exists("train_data"):
            train = tf.data.Dataset.load("train_data")
            dev = tf.data.Dataset.load("dev_data")
            print("Data loaded")
        else:
            train, dev = create_dataset("train"), create_dataset("dev")
            train.save("train_data")
            dev.save("dev_data")
            print("Data created")

        # robeczech.trainable = False
        model = Comprehender(args, robeczech, None)
        if args.model != "":
            model.load_weights(args.model)
        for layer in model.layers:
            layer.trainable = True
        lr = tf.keras.optimizers.schedules.CosineDecay(1e-6, len(train) * args.epochs)
        model.compile(
            optimizer=tf.optimizers.experimental.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9, jit_compile=False),
            loss={"a_start": tf.keras.losses.SparseCategoricalCrossentropy(), "a_end": tf.keras.losses.SparseCategoricalCrossentropy()},
            metrics={"a_start": [tf.metrics.SparseCategoricalAccuracy()], "a_end": [tf.metrics.SparseCategoricalAccuracy(name="accuracy_end")]})

        train = train.shuffle(len(train), seed=args.seed)
        train = train.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
        train = train.prefetch(tf.data.AUTOTUNE)  
        dev = dev.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
        dev = dev.prefetch(tf.data.AUTOTUNE)

        tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)

        def save_model(epoch, logs):
            model.save(os.path.join(args.logdir, f"ep{epoch + 1}.h5"), include_optimizer=False)

        model.fit(train, epochs=args.epochs, validation_data=dev,
                  callbacks=[tb_callback, tf.keras.callbacks.LambdaCallback(on_epoch_end=save_model)], verbose=2)

    else:
        args.logdir = "/".join(args.model.split("/")[:-1])
        model = Comprehender(args, robeczech)
        model.load_weights(args.model)

    if args.eval:
        test = create_tst_dataset(dataset="dev")
        filename = "reading_comprehension_val.txt"
    else:
        test = create_tst_dataset()
        filename = "reading_comprehension.txt"

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, filename), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the answers as strings, one per line.
        i = 0
        for inputs, helpers in zip(*test):
            out = model.predict(inputs)
            start = tf.argmax(out["a_start"][0]).numpy()
            end = tf.argmax(out["a_end"][0]).numpy()
            context, tok = helpers
            ch_start = tok.token_to_chars(start).start
            ch_end = tok.token_to_chars(end).end
            if ch_end > len(context):
                ch_end = len(context)
            answer = context[ch_start:ch_end]
            print(answer, file=predictions_file)
            i += 1
            print(f"{i} / {len(test[0])}")



if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
