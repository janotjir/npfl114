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
parser.add_argument("--iou_thr", default=0.5, type=float, help="IoU threshold for gold classes.")
parser.add_argument("--cls_balancing", default=False, action="store_true", help="Focal loss class balancing")
parser.add_argument("--alpha", default=0.25, type=float, help="Focal loss parameter")
parser.add_argument("--gamma", default=2, type=float, help="Focal loss parameter")


# Team members:
# 4c2c10df-00be-4008-8e01-1526b9225726
# dc535248-fa6c-4987-b49f-25b6ede7c87d


pyramid_scales = [8, 16, 32, 64, 128]
anchor_shapes = np.array([[1,1], [1,2], [2,1]])
input_shape = [320, 320]
A = len(pyramid_scales) * anchor_shapes.shape[0]


# TODO change anchor generating to match retinanet (5 pyramid scales)
class DetMuchNet(tf.keras.Model):
    def __init__(self):
        inputs = tf.keras.layers.Input(shape=[input_shape[0], input_shape[1], 3])

        c = self._backbone(inputs)
        p = self._pyramid_network(c)
        
        cls_out = self._classification_head(p)
        print(cls_out.shape)
        box_out = self._regression_head(p)
        print(box_out.shape)

        outputs = {
            "classes": cls_out,
            "boxes": box_out
        }

        super().__init__(inputs=inputs, outputs=outputs)

    def _backbone(self, inputs):
        backbone = tf.keras.applications.EfficientNetV2B0(include_top=False, weights="imagenet", input_shape=(input_shape[0], input_shape[1], 3))
        backbone = tf.keras.Model(
            inputs=backbone.input,
            outputs=[backbone.get_layer(layer).output for layer in [
                "top_activation", "block5e_add", "block3b_add"]]
        )
        backbone.trainable = False
        c5, c4, c3 = backbone(inputs)

        return [c5, c4, c3]

    def _pyramid_network(self, c):
        c5, c4, c3 = c
        p5 = tf.keras.layers.Conv2D(256, 1, padding='same')(c5)

        p4 = tf.keras.layers.Conv2D(256, 1, padding='same')(c4)
        p4 = tf.keras.layers.Add()([tf.keras.layers.UpSampling2D(2)(p5), p4])

        p3 = tf.keras.layers.Conv2D(256, 1, padding='same')(c3)
        p3 = tf.keras.layers.Add()([tf.keras.layers.UpSampling2D(2)(p4), p3])

        p5 = tf.keras.layers.Conv2D(256, 3, padding='same')(p5)
        p4 = tf.keras.layers.Conv2D(256, 3, padding='same')(p4)
        p3 = tf.keras.layers.Conv2D(256, 3, padding='same')(p3)
        p6 = tf.keras.layers.Conv2D(256, 3, strides=2, padding='same')(p5)
        p7 = tf.keras.layers.Conv2D(256, 3, strides=2, padding='same')(tf.keras.layers.Activation('relu')(p6))

        return [p3, p4, p5, p6, p7]

    def _classification_head(self, p):
        cls_head = tf.keras.Sequential()
        cls_head.add(tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'))
        cls_head.add(tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'))
        cls_head.add(tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'))
        cls_head.add(tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'))
        cls_head.add(tf.keras.layers.Conv2D(SVHN.LABELS*A, 3, activation=tf.nn.sigmoid, padding='same'))

        cls_out = []

        for _p in p:
            _p = cls_head(_p)
            _p = tf.reshape(_p, [args.batch_size, -1, SVHN.LABELS*A])
            cls_out.append(_p)

        cls_out = tf.concat(cls_out, axis=1)

        return cls_out

    def _regression_head(self, p):
        box_head = tf.keras.Sequential()
        box_head.add(tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'))
        box_head.add(tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'))
        box_head.add(tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'))
        box_head.add(tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'))
        box_head.add(tf.keras.layers.Conv2D(4*A, 3, activation=None, padding='same'))

        box_out = []

        for _p in p:
            _p = box_head(_p)
            _p = tf.reshape(_p, [args.batch_size, -1, 4*A])
            box_out.append(_p)

        box_out = tf.concat(box_out, axis=1)

        return box_out       
    

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
    anchor_cls = tf.one_hot(anchor_cls, 10)

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
    train = train.map(lambda img, cls, bbx: tf.py_function(prepare_examples, inp=[img, cls, bbx], Tout=[tf.uint8, tf.float32, tf.float32]))
    train = train.map(lambda img, cls, bbx: (img, {"classes": cls, "boxes": bbx}))

    dev = dev.map(lambda x: (x["image"], x["classes"], x['bboxes']))
    dev = dev.map(lambda img, cls, bbx: tf.py_function(prepare_examples, inp=[img, cls, bbx], Tout=[tf.uint8, tf.float32, tf.float32]))
    dev = dev.map(lambda img, cls, bbx: (img, {"classes": cls, "boxes": bbx}))

    train = train.shuffle(args.seed)
    train = train.batch(args.batch_size)
    train = train.prefetch(tf.data.AUTOTUNE)

    dev = dev.batch(args.batch_size)
    dev = dev.prefetch(tf.data.AUTOTUNE)
   
    #for img, lbl in train:
    #    print(img.shape, lbl["classes"].shape, lbl["boxes"].shape)

    # TODO: Create the model and train it
    model = DetMuchNet()
    model.compile(
        optimizer=tf.optimizers.experimental.AdamW(jit_compile=False),
        loss={
            "classes": tf.keras.losses.BinaryFocalCrossentropy(args.cls_balancing, args.alpha, args.gamma),
            "boxes": tf.keras.losses.MeanSquaredError(),
        },
        metrics={
            "classes": [tf.metrics.CategoricalAccuracy("accuracy")],
            #"boxes": [tf.metrics.BinaryAccuracy("accuracy")],
        },
    )

    logs = model.fit(train, epochs=args.epochs, validation_data=dev)

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
