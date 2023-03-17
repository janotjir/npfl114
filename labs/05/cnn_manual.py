#!/usr/bin/env python3
import argparse
import os
from typing import List, Tuple
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--cnn", default="5-3-2,10-3-2", type=str, help="CNN architecture.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--verify", default=True, action="store_true", help="Verify the implementation.")
# If you add more arguments, ReCodEx will keep them with your default values.


# Team members:
# 4c2c10df-00be-4008-8e01-1526b9225726
# dc535248-fa6c-4987-b49f-25b6ede7c87d


class Convolution:
    def __init__(self, filters: int, kernel_size: int, stride: int, input_shape: List[int], verify: bool) -> None:
        # Create a convolutional layer with the given arguments
        # and given input shape (e.g., [28, 28, 1]).
        self._filters = filters
        self._kernel_size = kernel_size
        self._stride = stride
        self._verify = verify

        # Here the kernel and bias variables are created
        self._kernel = tf.Variable(tf.initializers.GlorotUniform(seed=42)(
            [kernel_size, kernel_size, input_shape[2], filters]))
        self._bias = tf.Variable(tf.initializers.Zeros()([filters]))
        #self._bias = tf.Variable(tf.experimental.numpy.arange(0, filters, dtype=tf.float32))

    def forward(self, inputs: tf.Tensor) -> tf.Tensor:
        # TODO: Compute the forward propagation through the convolution
        # with `tf.nn.relu` activation, and return the result.
        #
        # In order for the computation to be reasonably fast, you cannot
        # manually iterate through the individual pixels, batch examples,
        # input filters, or output filters. However, you can manually
        # iterate through the kernel size.
        self._inshape = inputs.shape
        output = self._conv2d(inputs)
        output = tf.nn.relu(output)

        # If requested, verify that `output` contains a correct value.
        if self._verify:
            reference = tf.nn.relu(tf.nn.convolution(inputs, self._kernel, self._stride) + self._bias)
            np.testing.assert_allclose(output, reference, atol=1e-4, err_msg="Forward pass differs!")

        return output

    def _conv2d(self, inputs: tf.Tensor) -> tf.Tensor:
        patches = tf.image.extract_patches(images=inputs, sizes=[1, self._kernel_size, self._kernel_size, 1], strides=[1, self._stride, self._stride, 1], rates=[1, 1, 1, 1], padding='VALID')
        _, h, w, _ = patches.shape
        patches = tf.reshape(patches, (patches.shape[0], patches.shape[1]*patches.shape[2], patches.shape[3]))

        kernel = tf.reshape(self._kernel, (-1, self._filters))
        kernel = tf.expand_dims(kernel, axis=0)

        out = tf.matmul(patches, kernel) + self._bias
        out = tf.reshape(out, (out.shape[0], h, w, out.shape[-1]))

        return out

    def _get_inputs_gradient(self, outputs_gradient):
        b, hw, ww, f = outputs_gradient.shape
        #print(self._kernel.shape)
        #print(self._stride)
        #print(outputs_gradient.shape)
        prepared_inputs = np.zeros((b, hw+(hw-1)*(self._stride-1)+2*(self._kernel_size-1), ww+(ww-1)*(self._stride-1)+2*(self._kernel_size-1), f), dtype=np.float32)
        prepared_inputs[:, self._kernel_size-1:-self._kernel_size+1:self._stride, self._kernel_size-1:-self._kernel_size+1:self._stride, :] = outputs_gradient
        pad_right = self._inshape[2] - prepared_inputs.shape[2] + self._kernel_size - 1
        pad_bottom = self._inshape[1] - prepared_inputs.shape[1] + self._kernel_size - 1
        prepared_inputs = np.pad(prepared_inputs, ((0, 0), (0, pad_bottom), (0, pad_right), (0, 0)), mode='constant')
        prepared_inputs = tf.convert_to_tensor(prepared_inputs)

        patches = tf.image.extract_patches(images=prepared_inputs, sizes=[1, self._kernel_size, self._kernel_size, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')
        _, h, w, _ = patches.shape
        #print(patches.shape)
        patches = tf.reshape(patches, (patches.shape[0], patches.shape[1]*patches.shape[2], patches.shape[3]))

        kernel = self._kernel.numpy()
        kernel = tf.convert_to_tensor(np.transpose(np.transpose(kernel, (1, 0, 2, 3))[::-1], (1, 0, 2, 3))[::-1])
        kernel = tf.transpose(kernel, perm=[0, 1, 3, 2])
        kernel = tf.reshape(kernel, (-1, kernel.shape[-1]))
        kernel = tf.expand_dims(kernel, axis=0)

        '''kernel = tf.transpose(self._kernel, perm=[0, 1, 3, 2])
        kernel = tf.reshape(self._kernel, (-1, kernel.shape[-1]))
        kernel = tf.expand_dims(kernel, axis=0)'''

        out = tf.matmul(patches, kernel)
        out = tf.reshape(out, (out.shape[0], h, w, out.shape[-1]))
        #print(out.shape)

        return out

    def _get_kernel_gradient(self, inputs, outputs_gradient):
        '''_, kh, kw, f = outputs_gradient.shape
        print(kh, kw, self._stride)'''

        #print(inputs.shape, outputs_gradient.shape, self._stride, self._kernel.shape)
        b, hw, ww, f = outputs_gradient.shape
        kernel = np.zeros((b, hw*self._stride-1, ww*self._stride-1, f), dtype=np.float32)
        kernel[:, ::self._stride, ::self._stride, :] = outputs_gradient
        kernel = tf.convert_to_tensor(kernel)
        _, kh, kw, f = kernel.shape

        inputs = tf.transpose(inputs, perm=[3, 1, 2, 0])
        patches = tf.image.extract_patches(images=inputs, sizes=[1, kh, kw, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')
        _, h, w, _ = patches.shape
        patches = tf.reshape(patches, (patches.shape[0], patches.shape[1]*patches.shape[2], patches.shape[3]))

        kernel = tf.transpose(kernel, perm=[1, 2, 0, 3])
        kernel = tf.reshape(kernel, (-1, f))
        kernel = tf.expand_dims(kernel, axis=0)

        out = tf.matmul(patches, kernel)
        out = tf.reshape(out, (out.shape[0], h, w, out.shape[-1]))
        out = tf.transpose(out, perm=[1, 2, 0, 3])
        out = out[:self._kernel_size, :self._kernel_size, : , :]


        '''inputs = tf.transpose(inputs, perm=[3, 1, 2, 0])
        patches = tf.image.extract_patches(images=inputs, sizes=[1, kh, kw, 1], strides=[1, self._stride, self._stride, 1], rates=[1, 1, 1, 1], padding='VALID')
        _, h, w, _ = patches.shape
        print(patches.shape)
        patches = tf.reshape(patches, (patches.shape[0], patches.shape[1]*patches.shape[2], patches.shape[3]))

        outputs_gradient = tf.transpose(outputs_gradient, perm=[1, 2, 0, 3])
        outputs_gradient = tf.reshape(outputs_gradient, (-1, f))
        outputs_gradient = tf.expand_dims(outputs_gradient, axis=0)


        print(patches.shape, outputs_gradient.shape)

        out = tf.matmul(patches, outputs_gradient)
        out = tf.reshape(out, (out.shape[0], h, w, out.shape[-1]))
        out = tf.transpose(out, perm=[1, 2, 0, 3])'''

        return out

    def backward(
        self, inputs: tf.Tensor, outputs: tf.Tensor, outputs_gradient: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # TODO: Given this layer's inputs, this layer's outputs,
        # and the gradient with respect to the layer's outputs,
        # compute the derivatives of the loss with respect to
        # - the `inputs` layer,
        # - `self._kernel`,
        # - `self._bias`.

        # apply derivative of relu
        outputs_gradient = outputs_gradient.numpy()
        outputs_gradient[outputs <= 0] = 0
        outputs_gradient = tf.convert_to_tensor(outputs_gradient)

        # derivative wrt bias
        bias_gradient = tf.math.reduce_sum(tf.reshape(outputs_gradient, (-1, outputs_gradient.shape[-1])), axis=0)

        # derivative wrt input (full convolution of outputs_gradient and kernel that is rotated about 180 degrees)
        #print(inputs.shape)
        inputs_gradient = self._get_inputs_gradient(outputs_gradient)

        # derivative wrt kernel (convolution of outputs_gradient and inputs)
        kernel_gradient = self._get_kernel_gradient(inputs, outputs_gradient)

        # If requested, verify that the three computed gradients are correct.
        if self._verify:
            with tf.GradientTape() as tape:
                tape.watch(inputs)
                reference = tf.nn.relu(tf.nn.convolution(inputs, self._kernel, self._stride) + self._bias)
            for name, computed, reference in zip(
                    ["Inputs", "Kernel", "Bias"], [inputs_gradient, kernel_gradient, bias_gradient],
                    tape.gradient(reference, [inputs, self._kernel, self._bias], outputs_gradient)):
                np.testing.assert_allclose(computed, reference, atol=1e-4, err_msg=name + " gradient differs!")

        # Return the inputs gradient, the layer variables, and their gradients.
        return inputs_gradient, [self._kernel, self._bias], [kernel_gradient, bias_gradient]


class Model:
    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args

        # Create the convolutional layers according to `args.cnn`.
        input_shape = [MNIST.H, MNIST.W, MNIST.C]
        self._convs = []
        for layer in args.cnn.split(","):
            filters, kernel_size, stride = map(int, layer.split("-"))
            self._convs.append(Convolution(filters, kernel_size, stride, input_shape, args.verify))
            input_shape = [(input_shape[0] - kernel_size) // stride + 1,
                           (input_shape[1] - kernel_size) // stride + 1, filters]

        # Create the classification head
        self._flatten = tf.keras.layers.Flatten(input_shape=input_shape)
        self._classifier = tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax)

        # Create the metric and the optimizer
        self._accuracy = tf.metrics.SparseCategoricalAccuracy()
        self._optimizer = tf.optimizers.Adam(args.learning_rate, jit_compile=False)

    def train_epoch(self, dataset: MNIST.Dataset) -> None:
        for batch in dataset.batches(self._args.batch_size):
            # Forward pass through the convolutions
            hidden = tf.constant(batch["images"])
            conv_values = [hidden]
            for conv in self._convs:
                hidden = conv.forward(hidden)
                conv_values.append(hidden)

            # Run the classification head
            hidden_flat = self._flatten(hidden)
            predictions = self._classifier(hidden_flat)

            # Compute the gradients of the classifier and the convolution output
            d_logits = (predictions - tf.one_hot(batch["labels"], MNIST.LABELS)) / len(batch["images"])
            variables = [self._classifier.bias, self._classifier.kernel]
            gradients = [tf.reduce_sum(d_logits, 0), tf.linalg.matmul(hidden_flat, d_logits, transpose_a=True)]
            hidden_gradient = tf.reshape(tf.linalg.matvec(self._classifier.kernel, d_logits), hidden.shape)

            # Backpropagate the gradient through the convolutions
            for conv, inputs, outputs in reversed(list(zip(self._convs, conv_values[:-1], conv_values[1:]))):
                hidden_gradient, conv_variables, conv_gradients = conv.backward(inputs, outputs, hidden_gradient)
                variables.extend(conv_variables)
                gradients.extend(conv_gradients)

            # Update the weights
            self._optimizer.apply_gradients(zip(gradients, variables))

    def evaluate(self, dataset: MNIST.Dataset) -> float:
        self._accuracy.reset_states()
        for batch in dataset.batches(self._args.batch_size):
            hidden = batch["images"]
            for conv in self._convs:
                hidden = conv.forward(hidden)
            hidden = self._flatten(hidden)
            predictions = self._classifier(hidden)
            self._accuracy(batch["labels"], predictions)
        return self._accuracy.result()


def main(args: argparse.Namespace) -> float:
    # Set the random seed and the number of threads.
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.debug:
        tf.config.run_functions_eagerly(True)
        tf.data.experimental.enable_debug_mode()

    # Load data, using only 5 000 training images
    mnist = MNIST(size={"train": 5_000})

    # Create the model
    model = Model(args)

    for epoch in range(args.epochs):
        model.train_epoch(mnist.train)

        dev_accuracy = model.evaluate(mnist.dev)
        print("Dev accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * dev_accuracy))

    test_accuracy = model.evaluate(mnist.test)
    print("Test accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * test_accuracy))

    # Return dev and test accuracies for ReCodEx to validate.
    return dev_accuracy, test_accuracy


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)