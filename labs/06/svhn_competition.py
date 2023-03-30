#!/usr/bin/env python3
import argparse
import datetime
import os
import re
import sys
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import cv2

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
parser.add_argument("--cls_balancing", default=True, action="store_true", help="Focal loss class balancing")
parser.add_argument("--alpha", default=0.25, type=float, help="Focal loss parameter")
parser.add_argument("--gamma", default=2, type=float, help="Focal loss parameter")
parser.add_argument("--mask_loss", default=True, action='store_true', help="Set Huber loss to 0 for negative samples")
parser.add_argument("--resize", default=True, action='store_true', help="Instead of padding images with zeros, resize them")

parser.add_argument("--test", default=False, action="store_true", help="Load model and only annotate test data")
parser.add_argument("--eval", default=False, action="store_true", help="Load model and only annotate dev data")
parser.add_argument("--model", default="logs/pls/ep1.h5", type=str, help="Model path")
parser.add_argument("--anchors", default="logs/pls/anchors.npy", type=str, help="Anchors path")


# Team members:
# 4c2c10df-00be-4008-8e01-1526b9225726
# dc535248-fa6c-4987-b49f-25b6ede7c87d


pyramid_scales = [8, 16, 32, 64, 128]
anchor_shapes = np.array([[1,1], [2, 1]])
input_shape = [224, 224]
# input_shape = [320, 320]
A = anchor_shapes.shape[0]


class DetMuchNet(tf.keras.Model):
    def __init__(self, train_backbone=False):
        inputs = tf.keras.layers.Input(shape=[input_shape[0], input_shape[1], 3])

        c = self._backbone(inputs, train_backbone)
        p = self._pyramid_network(c)
        
        cls_out = self._classification_head(p)
        box_out = self._regression_head(p)

        outputs = {
            "classes": cls_out,
            "boxes": box_out
        }

        super().__init__(inputs=inputs, outputs=outputs)

    def _backbone(self, inputs, train_backbone):
        backbone = tf.keras.applications.EfficientNetV2B0(include_top=False, weights="imagenet", input_shape=(input_shape[0], input_shape[1], 3))
        backbone = tf.keras.Model(
            inputs=backbone.input,
            outputs=[backbone.get_layer(layer).output for layer in [
                "top_activation", "block5e_add", "block3b_add"]]
        )
        backbone.trainable = train_backbone
        c5, c4, c3 = backbone(inputs, training=False)   # training=False to freeze BatchNormalization layer

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
            _p = tf.reshape(_p, [-1, _p.shape[1]*_p.shape[2]*A, SVHN.LABELS])
            #_p = tf.reshape(_p, [args.batch_size, -1, SVHN.LABELS])
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
            _p = tf.reshape(_p, [-1, _p.shape[1]*_p.shape[2]*A, 4])
            #_p = tf.reshape(_p, [args.batch_size, -1, 4])
            box_out.append(_p)

        box_out = tf.concat(box_out, axis=1)
        #print(box_out.shape)

        return box_out       
    

class AnchorMaster:
    def __init__(self):
        anchors = np.array([], dtype=np.float32).reshape(0, 4*A)
        for scale in pyramid_scales:
            scaled_anchors = scale * anchor_shapes
            sc_anchors = np.array([], dtype=np.float32).reshape(int(np.ceil(input_shape[0]/scale)**2), 0)
            for anchor in scaled_anchors:
                x_positions = np.arange(input_shape[0]/scale)
                y_positions = np.arange(input_shape[1]/scale)
                xv, yv = np.meshgrid(x_positions, y_positions)
                upper_left = np.concatenate([yv[..., np.newaxis], xv[..., np.newaxis]], 2).reshape(-1, 2)
                upper_left *= scale
                lower_right = upper_left + anchor[np.newaxis, :]
                anchor_coords = np.hstack([upper_left, lower_right])
                print(sc_anchors.shape, anchor_coords.shape)
                sc_anchors = np.hstack([sc_anchors, anchor_coords])
            anchors = np.vstack([anchors, sc_anchors])
        anchors = anchors.reshape(anchors.shape[0]*A, 4)
        self.anchors = anchors
        np.set_printoptions(threshold=sys.maxsize)
        print(anchors)

    def save_anchors(self, dir):
        np.save(os.path.join(dir, "anchors.npy"), self.anchors)

    def load_anchors(self, path):
        self.anchors = np.load(path)

    def prepare_examples(self, img, cls, bbx):
        paddings = tf.constant([[0, input_shape[0] - img.shape[0]], [0, input_shape[1] - img.shape[1]], [0, 0]])
        img = tf.pad(img, paddings, mode='constant')

        anchor_cls, anchor_bbx = bboxes_utils.bboxes_training(self.anchors, cls.numpy(), bbx.numpy(), args.iou_thr)
        if args.mask_loss:
            anchor_bbx = np.hstack((anchor_bbx, tf.expand_dims(anchor_cls, axis=-1)))

        mask = anchor_cls > 0
        anchor_cls_new = np.zeros((anchor_cls.shape[0], SVHN.LABELS))
        anchor_cls_new[mask, :] = tf.one_hot(anchor_cls[mask]-1, SVHN.LABELS)

        return img, anchor_cls_new, anchor_bbx

    def prepare_examples_with_resize(self, img, cls, bbx):
        # get resize ratio
        shape = tf.cast(img.shape, tf.float32)
        ratio = input_shape[0] / tf.reduce_max(shape)
        
        # resize img and scale bboxes
        img = tf.image.resize_with_pad(img, input_shape[0], input_shape[1], antialias=True)
        img = tf.cast(img, tf.uint8)
        bbx = ratio * bbx

        # get anchors and their classes
        anchor_cls, anchor_bbx = bboxes_utils.bboxes_training(self.anchors, cls.numpy(), bbx.numpy(), args.iou_thr)
        if args.mask_loss:
            anchor_bbx = np.hstack((anchor_bbx, tf.expand_dims(anchor_cls, axis=-1)))

        mask = anchor_cls > 0
        anchor_cls_new = np.zeros((anchor_cls.shape[0], SVHN.LABELS))
        anchor_cls_new[mask, :] = tf.one_hot(anchor_cls[mask]-1, SVHN.LABELS)

        return img, anchor_cls_new, anchor_bbx, ratio

    def prepare_test_examples(self, img):
        paddings = tf.constant([[0, input_shape[0] - img.shape[0]], [0, input_shape[1] - img.shape[1]], [0, 0]])
        img = tf.pad(img, paddings, mode='constant')
        return img

    def prepare_test_examples_with_resize(self, img):
        # get resize ratio
        shape = tf.cast(img.shape, tf.float32)
        ratio = input_shape[0] / tf.reduce_max(shape)
        # resize img and scale bboxes
        img = tf.image.resize_with_pad(img, input_shape[0], input_shape[1], antialias=True)
        img = tf.cast(img, tf.uint8)
        return img, ratio

    
    def reconstruct_bboxes(self, bbx):
        return bboxes_utils.bboxes_from_fast_rcnn(self.anchors, bbx)


def masked_huber_loss(y_true, y_pred):
    loss = tf.keras.losses.Huber(name='masked_huber_loss')(y_true[:, :, :4], y_pred)
    loss = tf.where(tf.equal(y_true[:, :, -1] > 0, True), loss, 0.0)

    return loss


def decode_predictions(predicted_classes, predicted_bboxes, k=10, thresh=0.05, max_num_out=5):
    box_classes = tf.math.argmax(predicted_classes, axis=-1)
    box_confidence = tf.math.reduce_max(predicted_classes, axis=-1)
    
    # mask out bboxes with low confidence (less than 5%)
    mask = box_confidence > thresh
    box_classes = box_classes[mask]
    box_confidence = box_confidence[mask]
    predicted_bboxes = predicted_bboxes[mask]

    indices = tf.image.non_max_suppression(predicted_bboxes, box_confidence, max_num_out, iou_threshold=0.5, score_threshold=thresh)
    #indices = tf.image.combined_non_max_suppression(predicted_bboxes, predicted_classes, 2, 5, iou_threshold=0.5, score_threshold=0.05)

    predicted_classes = tf.gather(box_classes, indices)
    predicted_bboxes = tf.gather(predicted_bboxes, indices)

    return predicted_classes, predicted_bboxes


# visualize raw dataset
def visualize_dataset(data):

    def preprocess_data(img, classes, bboxes):
        shape = tf.cast(img.shape, tf.float32)
        ratio = input_shape[0] / tf.reduce_max(shape)
        img = tf.image.resize_with_pad(img, input_shape[0], input_shape[1], antialias=True)
        img = tf.cast(img, tf.uint8)
        bboxes = ratio * bboxes
        return img, classes, bboxes

    for img, label in data:
        img, label["classes"], label["boxes"] = preprocess_data(img, label["classes"], label["boxes"])
        img = img.numpy()
        for i in range(label["classes"].shape[0]):
            p1 = (int(label["boxes"][i, 1]), int(label["boxes"][i, 0]))
            p2 = (int(label["boxes"][i, 3]), int(label["boxes"][i, 2]))
            cv2.rectangle(img, p1, p2, (0,0,255), 1)
            cv2.putText(img, str(label["classes"][i].numpy()), (p1[0], p2[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 1)
        cv2.imshow("Data sample", img)
        cv2.waitKey(0)


# visualize prepared dataset
def visualize_data(data, am):
    for img, label in data:
        img = img.numpy()
        predicted_bboxes = am.reconstruct_bboxes(label["boxes"])
        for i in range(label["classes"].shape[0]):
            if np.sum(label["classes"][i]) > 0:
                p1 = (int(predicted_bboxes[i, 1]), int(predicted_bboxes[i, 0]))
                p2 = (int(predicted_bboxes[i, 3]), int(predicted_bboxes[i, 2]))
                cv2.rectangle(img, p1, p2, (0,0,255), 1)
                cv2.putText(img, str(np.squeeze(np.nonzero(label["classes"][i].numpy()))), (p1[0], p1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 1)
        cv2.imshow("Data sample", img)
        cv2.waitKey(0)


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
    os.makedirs(args.logdir, exist_ok=True)

    # Load the data
    svhn = SVHN()
    am = AnchorMaster()

    if not args.eval and not args.test:
        train = svhn.train
        dev = svhn.dev
        
        train = train.map(lambda x: (x["image"], x["classes"], x['bboxes']))
        if args.resize:
            train = train.map(lambda img, cls, bbx: tf.py_function(am.prepare_examples_with_resize, inp=[img, cls, bbx], Tout=[tf.uint8, tf.float32, tf.float32, tf.float32]))
            train = train.map(lambda img, cls, bbx, ratio: (img, {"classes": cls, "boxes": bbx}))
        else:
            train = train.map(lambda img, cls, bbx: tf.py_function(am.prepare_examples, inp=[img, cls, bbx], Tout=[tf.uint8, tf.float32, tf.float32]))
            train = train.map(lambda img, cls, bbx: (img, {"classes": cls, "boxes": bbx}))
        # visualize_data(train, am)

        dev = dev.map(lambda x: (x["image"], x["classes"], x['bboxes']))
        if args.resize:
            dev = dev.map(lambda img, cls, bbx: tf.py_function(am.prepare_examples_with_resize, inp=[img, cls, bbx], Tout=[tf.uint8, tf.float32, tf.float32, tf.float32]))
            dev = dev.map(lambda img, cls, bbx, ratio: (img, {"classes": cls, "boxes": bbx}))
        else:
            dev = dev.map(lambda img, cls, bbx: tf.py_function(am.prepare_examples, inp=[img, cls, bbx], Tout=[tf.uint8, tf.float32, tf.float32]))
            dev = dev.map(lambda img, cls, bbx: (img, {"classes": cls, "boxes": bbx}))
        # visualize_data(dev, am)

        train = train.shuffle(args.seed)
        train = train.batch(args.batch_size)
        train = train.prefetch(tf.data.AUTOTUNE)

        dev = dev.batch(args.batch_size)
        dev = dev.prefetch(tf.data.AUTOTUNE)

        am.save_anchors(args.logdir)

        # Create the model and train it
        model = DetMuchNet(train_backbone=True)
        if args.mask_loss:
            loss = masked_huber_loss
        else:
            loss = tf.keras.losses.Huber()
        model.compile(
            optimizer=tf.optimizers.experimental.AdamW(jit_compile=False),
            loss={
                "classes": tf.keras.losses.BinaryFocalCrossentropy(args.cls_balancing, args.alpha, args.gamma),
                "boxes": loss,
            },
            metrics={
                "classes": [tf.metrics.BinaryAccuracy("accuracy")],
            },
        )

        tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)
        def save_model(epoch, logs):
            model.save(os.path.join(args.logdir, f"ep{epoch+1}.h5"), include_optimizer=False)

        with tf.device("/GPU:0"):
            model.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[tb_callback, tf.keras.callbacks.LambdaCallback(on_epoch_end=save_model)])
    
    else:
        model = DetMuchNet(train_backbone=True)
        model.load_weights(args.model)
        am.load_anchors(args.anchors)
        args.logdir = "/".join(args.model.split("/")[:-1])

    if args.eval:
        test = svhn.dev
    else:
        test = svhn.test
    
    test = test.map(lambda x: (x["image"]))
    if args.resize:
        test = test.map(lambda img: tf.py_function(am.prepare_test_examples_with_resize, inp=[img], Tout=[tf.uint8, tf.float32]))
    else:
        test = test.map(lambda img: tf.py_function(am.prepare_test_examples, inp=[img], Tout=[tf.uint8]))
    test = test.batch(args.batch_size)

    # TODO evaluation in progress, just visual inspection of first batch for you my friend
    for batch in test:

        # ratios = tf.cast(batch[1], tf.float64)
        batch = batch[0]
        out = model.predict(batch)

        for j, p in enumerate(zip(out['classes'], out['boxes'])):
            predicted_classes, predicted_bboxes = p
            predicted_bboxes = am.reconstruct_bboxes(predicted_bboxes)
            #print(predicted_bboxes)
            predicted_classes, predicted_bboxes = decode_predictions(predicted_classes, predicted_bboxes)
            img = batch[j].numpy()
            for i in range(predicted_classes.shape[0]):
                #p1 = (int(predicted_bboxes[i, 1]/ratios[j]), int(predicted_bboxes[i, 0]/ratios[j]))
                #p2 = (int(predicted_bboxes[i, 3]/ratios[j]), int(predicted_bboxes[i, 2]/ratios[j]))
                p1 = (int(predicted_bboxes[i, 1]), int(predicted_bboxes[i, 0]))
                p2 = (int(predicted_bboxes[i, 3]), int(predicted_bboxes[i, 2]))
                cv2.rectangle(img, p1, p2, (0,0,255), 2)
                cv2.putText(img, str(predicted_classes[i].numpy()), (int(predicted_bboxes[i, 1]), int(predicted_bboxes[i, 0]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            cv2.imwrite("last_predict.png", img)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
        break

    exit()

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    with open(os.path.join(args.logdir, "svhn_competition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the digits and their bounding boxes on the test set.
        # Assume that for a single test image we get
        # - `predicted_classes`: a 1D array with the predicted digits,
        # - `predicted_bboxes`: a [len(predicted_classes), 4] array with bboxes;
        out = model.predict(test)
        for predicted_classes, predicted_bboxes in zip(out['classes'], out['boxes']):
            predicted_bboxes = am.reconstruct_bboxes(predicted_bboxes)
            # TODO: Apply nonmax supression
            output = []
            for label, bbox in zip(predicted_classes, predicted_bboxes):
                output += [label] + list(bbox)
                break # DELETE DIS PLS
            print(*output, file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
