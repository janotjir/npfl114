#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

import bboxes_utils
from svhn_dataset import SVHN

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--iou_thr", default=0.5, type=int, help="IoU threshold for gold classes.")


pyramid_scales = [2, 4, 8, 16]
anchor_shapes = np.array([[1,1], [1,2], [2,1]])
input_shape = [300, 300]


class KretinaNet(tf.keras.Model):
    def __init__(self):
        return
    

def prepare_examples(img, cls, bbx):
    paddings = tf.constant([[0, input_shape[0] - img.shape[0]], [0, input_shape[1] - img.shape[1]], [0, 0]])
    img = tf.pad(img, paddings, mode='constant')

    anchors = np.array([], dtype=np.float32).reshape(0, 4)
    for scale in pyramid_scales:
        scaled_anchors = scale * anchor_shapes
        for anchor in scaled_anchors:
            x_positions = np.arange(img.shape[0] - anchor[0] + 1)
            y_positions = np.arange(img.shape[1] - anchor[1] + 1)
            xv, yv = np.meshgrid(x_positions, y_positions)
            upper_left = np.concatenate([xv[..., np.newaxis], yv[..., np.newaxis]], 2).reshape(-1, 2)
            lower_right = upper_left + anchor[np.newaxis, :]
            anchor_coords = np.hstack([upper_left, lower_right])
            #print(anchor_coords)
            anchors = np.vstack([anchors, anchor_coords])
    
    #print(anchors)
    anchor_cls, anchor_bbx = bboxes_utils.bboxes_training(anchors, cls.numpy(), bbx.numpy(), args.iou_thr)

    return img, anchor_cls, anchor_bbx




def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    tf.keras.utils.set_random_seed(args.seed)
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

    # Load the data
    svhn = SVHN()

    train = svhn.train
    dev = svhn.dev
    test = svhn.test
    
    train = train.map(lambda x: (x["image"], x["classes"], x['bboxes']))
    train = train.map(lambda img, cls, bbx: tf.py_function(prepare_examples, inp=[img, cls, bbx], Tout=[tf.uint8, tf.int64, tf.float32]))
   
    for img, cls, bbx in train:
        print(img.shape, cls.shape, bbx.shape)

    # Load the EfficientNetV2-B0 model. It assumes the input images are
    # represented in [0-255] range using either `tf.uint8` or `tf.float32` type.
    backbone = tf.keras.applications.EfficientNetV2B0(include_top=False)

    # Extract features of different resolution. Assuming 224x224 input images
    # (you can set this explicitly via `input_shape` of the above constructor),
    # the below model returns five outputs with resolution 7x7, 14x14, 28x28, 56x56, 112x112.
    backbone = tf.keras.Model(
        inputs=backbone.input,
        outputs=[backbone.get_layer(layer).output for layer in [
             "top_activation", "block5e_add", "block3b_add", "block2b_add", "block1a_project_activation"]]
    )

    # TODO: Create the model and train it
    model = ...

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "svhn_competition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the digits and their bounding boxes on the test set.
        # Assume that for a single test image we get
        # - `predicted_classes`: a 1D array with the predicted digits,
        # - `predicted_bboxes`: a [len(predicted_classes), 4] array with bboxes;
        for predicted_classes, predicted_bboxes in ...:
            output = []
            for label, bbox in zip(predicted_classes, predicted_bboxes):
                output += [label] + list(bbox)
            print(*output, file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
