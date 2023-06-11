#!/usr/bin/env python3
import argparse
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import gymnasium as gym
import numpy as np
import tensorflow as tf
import datetime

import cart_pole_pixels_environment
import wrappers


# Team members:
# 4c2c10df-00be-4008-8e01-1526b9225726
# dc535248-fa6c-4987-b49f-25b6ede7c87d


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=500, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--evaluate", default=False, action="store_true", help="Just evaluate, no training")
parser.add_argument("--batch_size", default=1, type=int, help="Batch size.")
parser.add_argument("--episodes", default=2000, type=int, help="Training episodes.")
parser.add_argument("--learning_rate", default=1e-4, type=float, help="Learning rate.")
parser.add_argument("--model_path", default="", type=str, help="Specify model logdir")



class Agent:
    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        # TODO: Create a suitable model. The predict method assumes
        # the policy network is stored as `self._model`.
        #
        # Apart from the model defined in `reinforce`, define also another
        # model for computing the baseline (with a single output without an activation).
        # (Alternatively, this baseline computation can be grouped together
        # with the policy computation in a single `tf.keras.Model`.)
        #
        # Using Adam optimizer with given `args.learning_rate` for both models
        # is a good default.
        # raise NotImplementedError()
        inputs = tf.keras.layers.Input(shape=[80, 80, 3], dtype=tf.int8)
        inputs = tf.image.convert_image_dtype(inputs, tf.float32)
        
        # backbone
        hidden = tf.keras.layers.Conv2D(4, kernel_size=3, strides=2, padding='same', activation=None, use_bias=False)(inputs)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.keras.layers.Activation('relu')(hidden)

        hidden = tf.keras.layers.Conv2D(8, kernel_size=3, strides=2, padding='same', activation=None, use_bias=False)(hidden)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.keras.layers.Activation('relu')(hidden)

        hidden = tf.keras.layers.Conv2D(16, kernel_size=3, strides=2, padding='same', activation=None, use_bias=False)(hidden)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.keras.layers.Activation('relu')(hidden)

        hidden = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same', activation=None, use_bias=False)(hidden)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.keras.layers.Activation('relu')(hidden)

        #hidden = tf.keras.layers.MaxPooling2D()(hidden)
        #hidden = tf.keras.layers.Lambda(lambda z: tf.keras.backend.mean(z, [3]), name='reduce_mean')(hidden)
        #hidden = tf.keras.layers.Lambda(lambda z: tf.keras.backend.mean(z, [1, 2]), name='reduce_mean')(hidden)
        hidden = tf.keras.layers.Flatten()(hidden)
        
        # print(hidden.shape)
        # hidden = tf.keras.layers.Dense(32, activation='relu')(hidden)

        # heads
        action_out = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(hidden)
        baseline_out = tf.keras.layers.Dense(1, activation=None)(hidden)
        
        self._model = tf.keras.Model(inputs=inputs, outputs=action_out)
        self._model_baseline = tf.keras.Model(inputs=inputs, outputs=baseline_out)

        self._model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate), loss=tf.keras.losses.SparseCategoricalCrossentropy())
        self._model_baseline.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate), loss=tf.keras.losses.MeanSquaredError())

    # Define a training method.
    #
    # Note that we need to use `raw_tf_function` (a faster variant of `tf.function`)
    # and manual `tf.GradientTape` for efficiency (using `fit` or `train_on_batch`
    # on extremely small batches has considerable overhead).
    @wrappers.raw_tf_function(dynamic_dims=1)
    def train(self, states: np.ndarray, actions: np.ndarray, returns: np.ndarray) -> None:
        # TODO: Perform training, using the loss from the REINFORCE with baseline
        # algorithm. You should:
        # - compute the predicted baseline using the baseline model
        # - train the baseline model to predict `returns`
        # - train the policy model, using `returns - predicted_baseline` as
        #   the advantage estimate
        # raise NotImplementedError()
        with tf.GradientTape() as tape:
            predicted_baseline = tf.squeeze(tf.cast(self._model_baseline(states), tf.float64))
            baseline_loss = self._model_baseline.compiled_loss(y_true=returns, y_pred=predicted_baseline)
        
        grads_baseline = tape.gradient(baseline_loss, self._model_baseline.trainable_variables)
        self._model_baseline.optimizer.apply_gradients(zip(grads_baseline, self._model_baseline.trainable_variables))

        with tf.GradientTape() as tape:
            preds = self._model(states)
            weights = returns- predicted_baseline
            loss = self._model.compiled_loss(y_true=actions, y_pred=preds, sample_weight=weights)
        
        grads = tape.gradient(loss, self._model.trainable_variables)
        self._model.optimizer.apply_gradients(zip(grads, self._model.trainable_variables))

    # Predict method, again with the `raw_tf_function` for efficiency.
    @wrappers.raw_tf_function(dynamic_dims=1)
    def predict(self, states: np.ndarray) -> np.ndarray:
        return self._model(states)


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    if args.seed is not None:
        tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.debug:
        tf.config.run_functions_eagerly(True)
        # tf.data.experimental.enable_debug_mode()

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    ))
    os.makedirs(args.logdir, exist_ok=True)

    if not args.recodex and not args.evaluate:
        # TODO: Perform training
        os.makedirs(args.logdir, exist_ok=True)
        
        # Construct the agent
        agent = Agent(env, args)
        if args.model_path != "":
            agent._model.load_weights(os.path.join(args.model_path, "model_adam.h5"))
            agent._model_baseline.load_weights(os.path.join(args.model_path, "baseline_adam.h5"))

        # Training
        for _ in range(args.episodes // args.batch_size):
            batch_states, batch_actions, batch_returns = [], [], []
            for _ in range(args.batch_size):
                # Perform episode
                states, actions, rewards = [], [], []
                state, done = env.reset()[0], False
                while not done:
                    # TODO(reinforce): Choose `action` according to probabilities
                    # distribution (see `np.random.choice`), which you
                    # can compute using `agent.predict` and current `state`.
                    probabilities = agent.predict(tf.expand_dims(state, axis=0))[0]
                    action = np.random.choice(2, p=probabilities)

                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)

                    state = next_state

                # TODO(reinforce): Compute returns from the received rewards
                returns = [np.sum(rewards[i:]) for i in range(len(rewards))]

                # TODO(reinforce): Add states, actions and returns to the training batch
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_returns.extend(returns)

            # TODO(reinforce): Train using the generated batch.
            agent.train(batch_states, batch_actions, batch_returns)

        # TODO: save trained model
        agent._model.save_weights(os.path.join(args.logdir, "model_adam.h5"))
        agent._model_baseline.save_weights(os.path.join(args.logdir, "baseline_adam.h5"))
       
        
    else:
        # TODO: Load a pre-trained agent and evaluate it.
        agent = Agent(env, args)
        agent._model.load_weights("model_adam.h5")
        agent._model_baseline.load_weights("baseline_adam.h5")
        pass
    
    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            # TODO(reinforce): Choose a greedy action
            action = agent.predict(tf.expand_dims(state, axis=0))[0]
            action = tf.math.argmax(action).numpy()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make("CartPolePixels-v1"), args.seed, args.render_each)

    main(env, args)
