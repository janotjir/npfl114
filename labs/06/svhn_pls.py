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

import cv2

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--iou_gold", default=0.5, type=float, help="IoU threshold for gold classes.")

parser.add_argument("--test", default=False, action="store_true", help="Load model and only annotate test data")
parser.add_argument("--eval", default=False, action="store_true", help="Load model and only annotate dev data")
parser.add_argument("--model", default="", type=str, help="Model path")
parser.add_argument("--anchors", default="", type=str, help="Anchors path")
parser.add_argument("--visualize", default=False, action="store_true", help="Visualize predictions = store first batch of predictions to results folder")


# Team members:
# 4c2c10df-00be-4008-8e01-1526b9225726
# dc535248-fa6c-4987-b49f-25b6ede7c87d


pyramid_scales = [8, 16, 32, 64, 128]
anchor_shapes = np.array([[1., 1.], [2., 1.], [1.*2**(1/3), 1.*2**(1/3)], [2.*2**(1/3), 1.*2**(1/3)], [1.*2**(2/3), 1.*2**(2/3)], [2.*2**(2/3), 1.*2**(2/3)]])
scales = 1
input_shape = [224, 224]
A = anchor_shapes.shape[0] * scales


def focal_loss(alpha=0.25, gamma=2.0):

    def _call(y_true, y_pred):
        labels = y_true[:, :, :-1]
        positive_mask = y_true[:, :, -1]

        # calculate loss, without reduction
        loss = tf.keras.losses.BinaryFocalCrossentropy(alpha=alpha, gamma=gamma, reduction=tf.keras.losses.Reduction.NONE)(labels, y_pred)

        # calculate coeff for normalization as number of positive samples
        norm_coeff = tf.where(tf.keras.backend.equal(positive_mask, 1))
        norm_coeff = tf.keras.backend.cast(tf.keras.backend.shape(norm_coeff)[0], tf.keras.backend.floatx())
        norm_coeff = tf.keras.backend.maximum(tf.keras.backend.cast_to_floatx(1.0), norm_coeff)
        
        return tf.keras.backend.sum(loss) / norm_coeff

    return _call

def smooth_l1_loss(sigma=1.0):
    pass


def focal(alpha=0.25, gamma=2.0, cutoff=0.5):
    '''
    adapted from https://github.com/fizyr/keras-retinanet.git
    '''
    
    def _focal(y_true, y_pred):
        labels         = y_true[:, :, :-1]
        anchor_state   = y_true[:, :, -1]  # 0 for background, 1 for object
        classification = y_pred

        # compute the focal loss
        alpha_factor = tf.keras.backend.ones_like(labels) * alpha
        alpha_factor = tf.where(tf.keras.backend.greater(labels, cutoff), alpha_factor, 1 - alpha_factor)
        focal_weight = tf.where(tf.keras.backend.greater(labels, cutoff), 1 - classification, classification)
        focal_weight = alpha_factor * focal_weight ** gamma

        # print(labels.shape, classification.shape)
        cls_loss = focal_weight * tf.keras.backend.binary_crossentropy(labels, classification)

        # compute the normalizer: the number of positive anchors
        normalizer = tf.where(tf.keras.backend.equal(anchor_state, 1))
        normalizer = tf.keras.backend.cast(tf.keras.backend.shape(normalizer)[0], tf.keras.backend.floatx())
        normalizer = tf.keras.backend.maximum(tf.keras.backend.cast_to_floatx(1.0), normalizer)

        return tf.keras.backend.sum(cls_loss) / normalizer

    return _focal

def smooth_l1(sigma=3.0):
    '''
    adapted from https://github.com/fizyr/keras-retinanet.git
    '''

    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        # separate target and state
        regression        = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state      = y_true[:, :, -1]

        # filter out negative anchors
        indices           = tf.where(tf.keras.backend.equal(anchor_state, 1))
        regression        = tf.gather_nd(regression, indices)
        regression_target = tf.gather_nd(regression_target, indices)

        regression_diff = regression - regression_target
        regression_diff = tf.keras.backend.abs(regression_diff)
        regression_loss = tf.where(
            tf.keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * tf.keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # compute the normalizer: the number of positive anchors
        normalizer = tf.keras.backend.maximum(1, tf.keras.backend.shape(indices)[0])
        normalizer = tf.keras.backend.cast(normalizer, dtype=tf.keras.backend.floatx())
        return tf.keras.backend.sum(regression_loss) / normalizer

    return _smooth_l1


class RetinaNet(tf.keras.Model):
    def __init__(self, train_backbone=False):
        inputs = tf.keras.layers.Input(shape=[input_shape[0], input_shape[1], 3])

        c = self._backbone(inputs, train_backbone)
        p = self._pyramid_network(c)
        
        cls_out = self._classification_head(p)
        box_out = self._regression_head(p)

        outputs = {"classes": cls_out, "boxes": box_out}
        super().__init__(inputs=inputs, outputs=outputs)

    def _backbone(self, inputs, train_backbone):
        backbone = tf.keras.applications.EfficientNetV2B0(include_top=False, weights="imagenet", input_shape=(input_shape[0], input_shape[1], 3))
        backbone = tf.keras.Model(
            inputs=backbone.input,
            outputs=[backbone.get_layer(layer).output for layer in [
                "top_activation", "block5e_add", "block3b_add"]]
        )
        backbone.trainable = train_backbone
        c5, c4, c3 = backbone(inputs, training=False) 

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
            box_out.append(_p)

        box_out = tf.concat(box_out, axis=1)

        return box_out       


class AnchorDataMaster:
    def __init__(self):
        self.pyramid_scales = pyramid_scales
        self.img_size = input_shape
        self.scales = scales
        self.aspect_ratios = anchor_shapes
        #self.generate_anchors(visualize=False)
        self.generate_anchors_()

    def generate_anchors_(self, visualize=False):
        anchors = np.array([], dtype=np.float32).reshape(0, 4*A)
        for scale in self.pyramid_scales:
            scaled_anchors = scale * self.aspect_ratios
            sc_anchors = np.array([], dtype=np.float32).reshape(int(np.ceil(self.img_size[0]/scale)**2), 0)
            for anchor in scaled_anchors:
                x_positions = np.arange(input_shape[0]/scale)
                y_positions = np.arange(input_shape[1]/scale)
                xv, yv = np.meshgrid(x_positions, y_positions)
                upper_left = np.concatenate([yv[..., np.newaxis], xv[..., np.newaxis]], 2).reshape(-1, 2)
                upper_left *= scale
                lower_right = upper_left + anchor[np.newaxis, :]
                anchor_coords = np.hstack([upper_left, lower_right])
                #print(sc_anchors.shape, anchor_coords.shape)
                sc_anchors = np.hstack([sc_anchors, anchor_coords])
            anchors = np.vstack([anchors, sc_anchors])
        anchors = anchors.reshape(anchors.shape[0]*A, 4)
        self.anchors = anchors

        if visualize:
            for i in range(self.anchors.shape[0]):
                    img = np.zeros((self.img_size[0],self.img_size[1],3))
                    p1 = (int(self.anchors[i, 1]), int(self.anchors[i, 0]))
                    p2 = (int(self.anchors[i, 3]), int(self.anchors[i, 2]))
                    cv2.rectangle(img, p1, p2, (0,0,255), 1)
                    cv2.imshow("Anchors", img)
                    cv2.waitKey(0)

    def generate_anchors(self, visualize=False):
        self.anchors = np.empty((0, 4))

        shapes = []
        for scale in range(self.scales):
            for shape in self.aspect_ratios:
                shapes.append(shape * 2**(scale/self.scales))

        for stride in self.pyramid_scales:
            act_level_anchors = []
            for asp_h, asp_w in shapes:
                centers_x = np.arange(stride / 2, self.img_size[0], stride)
                centers_y = np.arange(stride / 2, self.img_size[1], stride)
                xv, yv = np.meshgrid(centers_x, centers_y)
                xv = xv.reshape(-1)
                yv = yv.reshape(-1)

                anchor_h = stride * asp_h / 2
                anchor_w = stride * asp_w / 2

                boxes = np.vstack((yv - anchor_h, xv - anchor_w, yv + anchor_h, xv + anchor_w)).T
                act_level_anchors.append(np.expand_dims(boxes, axis=1))
            
            act_level_anchors = np.concatenate(act_level_anchors, axis=1).reshape(-1, 4)
            self.anchors = np.vstack((self.anchors, act_level_anchors))
        
        if visualize:
            for i in range(self.anchors.shape[0]):
                img = np.zeros((self.img_size[0],self.img_size[1],3))
                p1 = (int(self.anchors[i, 1]), int(self.anchors[i, 0]))
                p2 = (int(self.anchors[i, 3]), int(self.anchors[i, 2]))
                cv2.rectangle(img, p1, p2, (0,0,255), 1)
                cv2.imshow("Anchors", img)
                cv2.waitKey(0)

    def save_anchors(self, dir):
        np.save(os.path.join(dir, "anchors.npy"), self.anchors)

    def load_anchors(self, path):
        self.anchors = np.load(path)

    def prepare_examples_with_resize(self, img, classes, bboxes):
        # get resize ratio
        shape = tf.cast(img.shape, tf.float32)
        ratio = input_shape[0] / tf.reduce_max(shape)
        
        # resize img and scale bboxes
        img = tf.image.resize_with_pad(img, input_shape[0], input_shape[1], antialias=True)
        img = tf.cast(img, tf.uint8)
        bboxes = ratio * bboxes

        # get anchors and their classes
        anchor_classes, anchor_bboxes = bboxes_utils.bboxes_training(self.anchors, classes.numpy(), bboxes.numpy(), args.iou_gold)

        positive_mask = np.expand_dims(anchor_classes > 0, axis=-1)
        anchor_bboxes = np.hstack((anchor_bboxes, positive_mask))
        anchor_classes = tf.one_hot(anchor_classes-1, SVHN.LABELS) * positive_mask
        anchor_classes = np.hstack((anchor_classes, positive_mask))

        '''img = img.numpy()
        for i in range(anchor_classes.shape[0]):
            if any(anchor_classes[i] > 0):
                print(anchor_classes[i])
                p1 = (int(self.anchors[i, 1]), int(self.anchors[i, 0]))
                p2 = (int(self.anchors[i, 3]), int(self.anchors[i, 2]))
                cv2.rectangle(img, p1, p2, (0,0,255), 1)
                cv2.imshow("Anchors", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''

        return img, anchor_classes, anchor_bboxes

    def prepare_test_examples_with_resize(self, img):
        # get resize ratio
        shape = tf.cast(img.shape, tf.float32)
        ratio = input_shape[0] / tf.reduce_max(shape)

        # resize img and scale bboxes
        img = tf.image.resize_with_pad(img, input_shape[0], input_shape[1], antialias=True)
        img = tf.cast(img, tf.uint8)

        '''cv2.imshow("sample", img.numpy())
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''

        return img, ratio

    def reconstruct_bboxes(self, bbx):
        return bboxes_utils.bboxes_from_fast_rcnn(self.anchors, bbx)

    def prepare_data(self, data, training=True, shuffle=False):
        if training:
            data = data.map(lambda x: (x["image"], x["classes"], x['bboxes']))
            data = data.map(lambda img, cls, bbx: tf.py_function(self.prepare_examples_with_resize, inp=[img, cls, bbx], Tout=[tf.uint8, tf.float32, tf.float32]))
            data = data.map(lambda img, cls, bbx: (img, {"classes": cls, "boxes": bbx}))
            if shuffle:
                data = data.shuffle(args.seed)
            data = data.batch(args.batch_size)
            data = data.prefetch(tf.data.AUTOTUNE)
        else:
            data = data.map(lambda x: (x["image"]))
            data = data.map(lambda img: tf.py_function(self.prepare_test_examples_with_resize, inp=[img], Tout=[tf.uint8, tf.float32]))
            data = data.batch(args.batch_size)

        return data

    def decode_predictions(self, pred_classes, pred_bboxes, num_out=4, thresh=0.2, iou_thresh=0.25):
        scores = tf.math.reduce_max(pred_classes, axis=-1)
        pred_classes = tf.math.argmax(pred_classes, axis=-1)
        pred_bboxes = self.reconstruct_bboxes(pred_bboxes)
        
        indices = tf.image.non_max_suppression(pred_bboxes, scores, num_out, iou_threshold=iou_thresh, score_threshold=thresh)

        pred_bboxes = tf.gather(pred_bboxes, indices)
        pred_classes = tf.gather(pred_classes, indices)
        scores = tf.gather(scores, indices)

        return pred_classes, pred_bboxes, scores


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.debug:
        tf.config.run_functions_eagerly(True)
        tf.data.experimental.enable_debug_mode()

    # Load and prepare the data
    svhn = SVHN()
    am = AnchorDataMaster()

    # TRAINING
    if not args.test and not args.eval:
        # Create logdir name
        args.logdir = os.path.join("logs", "{}-{}-{}".format(
            os.path.basename(globals().get("__file__", "notebook")),
            datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
            ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
        ))
        os.makedirs(args.logdir, exist_ok=True)

        train = am.prepare_data(svhn.train, training=True, shuffle=True)
        dev = am.prepare_data(svhn.dev, training=True)
        am.save_anchors(args.logdir)

        # Create the model and train it
        model = RetinaNet(train_backbone=True)
        if args.model != "":
            model.load_weights(args.model)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, clipnorm=0.001, jit_compile=False),
            loss={
                'classes': focal(),
                'boxes': smooth_l1()
            }
        )

        tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)
        def save_model(epoch, logs):
            model.save(os.path.join(args.logdir, f"ep{epoch+1}.h5"), include_optimizer=False)

        with tf.device("/GPU:0"):
            model.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[tb_callback, tf.keras.callbacks.LambdaCallback(on_epoch_end=save_model)])
    else:
        model = RetinaNet(train_backbone=True)
        model.load_weights(args.model)
        am.load_anchors(args.anchors)
        args.logdir = "/".join(args.model.split("/")[:-1])

    if args.eval:
        test = am.prepare_data(svhn.dev, training=False)
        filename = "svhn_competition_val.txt"
    else:
        test = am.prepare_data(svhn.test, training=False)
        filename = "svhn_competition.txt"

    count = 0
    with open(os.path.join(args.logdir, filename), "w", encoding="utf-8") as predictions_file:
        for batch in test:
            ratios = tf.cast(batch[1], tf.float64)
            imgs = batch[0]
            out = model.predict(imgs)

            for i, pred in enumerate(zip(out['classes'], out['boxes'])):
                pred_classes, pred_bboxes = pred
                pred_classes, pred_bboxes, scores = am.decode_predictions(pred_classes, pred_bboxes)

                if args.visualize:
                    img = imgs[i].numpy()
                
                output = []
                for j in range(pred_bboxes.shape[0]):
                    output += [pred_classes[j].numpy()] + list((pred_bboxes[j] / ratios[i]).numpy())
                    
                    if args.visualize:
                        p1 = (int(pred_bboxes[j, 1]), int(pred_bboxes[j, 0]))
                        p2 = (int(pred_bboxes[j, 3]), int(pred_bboxes[j, 2]))
                        cv2.rectangle(img, p1, p2, (0,0,255), 2)
                        cv2.putText(img, f"{pred_classes[j].numpy()}:{scores[j].numpy():.2f}", (int(pred_bboxes[j, 1]), int(pred_bboxes[j, 0]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

                print(*output, file=predictions_file)

                if args.visualize and count < args.batch_size:
                    os.makedirs("results", exist_ok=True)
                    cv2.imwrite(f"results/{count}.png", img)
                    count += 1


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

