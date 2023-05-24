#!/usr/bin/env python3
import argparse
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import gymnasium as gym
import numpy as np
import tensorflow as tf

import wrappers


# Team members:
# 4c2c10df-00be-4008-8e01-1526b9225726
# dc535248-fa6c-4987-b49f-25b6ede7c87d


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=1, type=int, help="Batch size.")
parser.add_argument("--episodes", default=200, type=int, help="Training episodes.")
parser.add_argument("--hidden_layer_size", default=128, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate.")


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
        '''input = tf.keras.layers.Input(shape=(4,))
        common = tf.keras.layers.Dense(args.hidden_layer_size, activation=None)(input)
        
        action_out = tf.keras.layers.Activation('tanh')(common)
        action_out = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(action_out)
        self._model = tf.keras.Model(inputs=input, outputs=action_out)

        baseline_out = tf.keras.layers.Activation('relu')(common)
        baseline_out = tf.keras.layers.Dense(1, activation=None)(baseline_out)
        self._model_baseline = tf.keras.Model(inputs=input, outputs=baseline_out)'''

        self._model = tf.keras.Sequential()
        self._model.add(tf.keras.layers.Input(shape=(4,)))
        self._model.add(tf.keras.layers.Dense(args.hidden_layer_size, activation='tanh', input_shape=[4]))
        self._model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))
        

        self._model_baseline = tf.keras.Sequential()
        self._model_baseline.add(tf.keras.layers.Input(shape=(4,)))
        self._model_baseline.add(tf.keras.layers.Dense(128, activation='relu', input_shape=[4]))
        self._model_baseline.add(tf.keras.layers.Dense(1, activation=None))

        self._model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate), 
                            loss=tf.keras.losses.SparseCategoricalCrossentropy())

        self._model_baseline.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                                    loss=tf.keras.losses.MeanSquaredError())

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

    # Construct the agent
    agent = Agent(env, args)

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

    # Final evaluation
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
    env = wrappers.EvaluationEnv(gym.make("CartPole-v1"), args.seed, args.render_each)

    main(env, args)
