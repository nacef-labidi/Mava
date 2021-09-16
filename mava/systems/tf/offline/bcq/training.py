# python3
# Copyright 2021 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Discreet Batch constrained Q-learning trainer implementation.

This implementation is like the one in Acme BCQ.
"""

import copy
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import reverb
import sonnet as snt
import tensorflow as tf
import trfl
from acme.tf import utils as tf2_utils
from acme.types import NestedArray
from acme.tf import losses
from acme.utils import counting, loggers

import mava
from mava import types as mava_types
from mava.systems.tf import savers as tf2_savers
from mava.utils import training_utils as train_utils

train_utils.set_growing_gpu_memory()


class BCQTrainer(mava.Trainer):
    """Discreet Batch Constrained Q-learnig trainer.

    This is the trainer component of an BCQ system. IE it takes a dataset
    as input and implements update functionality to learn from this dataset.
    """

    def __init__(
        self,
        agents: List[str],
        agent_types: List[str],
        q_networks: Dict[str, snt.Module],
        g_networks: Dict[str, snt.Module],
        threshold: float,
        dataset: tf.data.Dataset,
        agent_net_keys: Dict[str, str],
        learning_rate: float = 1e-3,
        discount: float = 0.99,
        huber_loss_parameter: float = 1.,
        target_update_period: int = 100,
        max_gradient_norm: float = None,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        checkpoint: bool = True,
        checkpoint_subpath: str = "~/mava/",
        checkpoint_minute_interval: int = 15,
    ):
        """Initialise the trainer.

        Args:
            agents: agent ids, e.g. "agent_0".
            agent_types: agent types, e.g. "speaker" or "listener".
            q_networks: q-networks being optimized.
            g_networks: behavior cloning networks being optimized.
            dataset: training dataset.
            learning_rate: learning rate of the network optimizers.
            agent_net_keys: specifies what network each agent uses. 
                Defaults to {}.
            discount: agent discount. Defaults to 0.99.
            target_update_period: learner steps between target network update.
                Defaults to 100.
            max_gradient_norm: maximum allowed norm for gradients
                before clipping is applied. Defaults to None.
            counter: step counter object. Defaults to None.
            logger: logger object for logging trainer statistics. 
                Defaults to None.
            checkpoint: whether to checkpoint networks. Defaults to True.
            checkpoint_subpath: subdirectory for storing checkpoints.
                Defaults to "~/mava/".
            checkpoint_minute_interval:  time in minutes between checkpoints.
        """
        self._agents = agents
        self._agent_types = agent_types
        self._agent_net_keys = agent_net_keys
        self._checkpoint = checkpoint

        # Internalise the hyperparameters.
        self._discount = discount
        self._target_update_period = target_update_period
        self._huber_loss_parameter = huber_loss_parameter

        # Store q-networks
        self._q_networks = q_networks
        self._target_q_networks = {}
        for key, value in q_networks.items():
            self._target_q_networks[key] = copy.deepcopy(value)

        # Store g-networks
        self._g_networks = g_networks

        # Store threshold
        assert threshold >= 0 and threshold <= 1
        self._threshold = threshold

        # General trainer book-keeping and loggers.
        self._counter = counter or counting.Counter()
        self._logger = logger

        # Set up gradient clipping.
        if max_gradient_norm is not None:
            self._max_gradient_norm = tf.convert_to_tensor(max_gradient_norm)
        else:  # A very large number. Infinity can result in NaNs.
            self._max_gradient_norm = tf.convert_to_tensor(1e10)

        # Necessary to track when to update target networks.
        self._num_steps = tf.Variable(0, dtype=tf.int32)

        # Create an iterator to go through the dataset.
        self._iterator = dataset

        # Dictionary with network keys for each agent.
        self.unique_net_keys = set(self._agent_net_keys.values())

        # Create optimizers for different networks.
        self._q_optimizers, self._g_optimizers = {}, {}
        for key in self.unique_net_keys:
            self._q_optimizers[key] = snt.optimizers.Adam(learning_rate)
            self._g_optimizers[key] = snt.optimizers.Adam(learning_rate)

        # Expose the variables.
        self._system_network_variables: Dict[str, Dict[str, snt.Module]] = {
            "q_network": {},
            "g_network": {}
        }
        for agent_key in self.unique_net_keys:
            q_network_to_expose = self._q_networks[agent_key]
            g_network_to_expose = self._g_networks[agent_key]

            self._system_network_variables["q_network"][
                agent_key
            ] = q_network_to_expose.variables

            self._system_network_variables["g_network"][
                agent_key
            ] = g_network_to_expose.variables

        # Checkpointer
        self._system_checkpointer = {}
        if checkpoint:
            for agent_key in self.unique_net_keys:
                checkpointer = tf2_savers.Checkpointer(
                    directory=checkpoint_subpath,
                    time_delta_minutes=checkpoint_minute_interval,
                    objects_to_save={
                        "counter": self._counter,
                        "q_network": self._q_networks[agent_key],
                        "g_network": self._g_networks[agent_key],
                        "q_optimizer": self._q_optimizers[agent_key],
                        "g_optimizer": self._g_optimizers[agent_key],
                        "num_steps": self._num_steps,
                    },
                    enable_checkpointing=checkpoint,
                )

                self._system_checkpointer[agent_key] = checkpointer

        # Do not record timestamps until after the first learning step is done.
        self._timestamp: Optional[float] = None

    def _get_feed_by_network_type(
        self, 
        observations, 
        next_observations, 
        actions, 
        rewards, 
        discounts
    ) -> Tuple:
        # Initalize dicts
        net_observation = {}
        net_next_observation = {}
        net_action = {}
        net_reward = {}
        net_discount = {}
        for agent in self._agents:
            agent_type = self._agent_net_keys[agent]

            # Concat observations
            if agent_type not in net_observation.keys():
                net_observation[agent_type] = observations[agent].observation
            else:
                net_observation[agent_type] = tf.concat(
                    [
                        net_observation[agent_type],
                        observations[agent].observation
                    ],
                    axis=0
                )

            # Concat next observations
            if agent_type not in net_next_observation.keys():
                net_next_observation[agent_type] = next_observations[agent].observation
            else:
                net_next_observation[agent_type] = tf.concat(
                    [
                        net_next_observation[agent_type],
                        next_observations[agent].observation
                    ],
                    axis=0
                )

            # Concat actions
            if agent_type not in net_action.keys():
                net_action[agent_type] = actions[agent]
            else:
                net_action[agent_type] = tf.concat(
                    [
                       net_action[agent_type],
                        actions[agent]
                    ],
                    axis=0
                )

            # Concat rewards
            if agent_type not in net_reward.keys():
                net_reward[agent_type] = rewards[agent]
            else:
                net_reward[agent_type] = tf.concat(
                    [
                       net_reward[agent_type],
                        rewards[agent]
                    ],
                    axis=0
                )

            # Concat discounts
            if agent_type not in net_discount.keys():
                net_discount[agent_type] = discounts[agent]
            else:
                net_discount[agent_type] = tf.concat(
                    [
                       net_discount[agent_type],
                        discounts[agent]
                    ],
                    axis=0
                )

        return (net_observation, net_next_observation, 
            net_action, net_reward, net_discount)

    def _filtered_q_value(self, net_key, observation):
        q_t = self._q_networks[net_key](observation)
        g_t = tf.nn.softmax(self._g_networks[net_key](observation))
        normalized_g_t = g_t / tf.reduce_max(g_t, axis=-1, keepdims=True)

        # Filter actions based on g-network outputs.
        min_q = tf.reduce_min(q_t, axis=-1, keepdims=True)
        return tf.where(normalized_g_t >= self._threshold, q_t, min_q)

    def _update_target_networks(self) -> None:
        """Sync the target network parameters with the latest online network
        parameters"""

        for key in self.unique_net_keys:
            # Update target network.
            online_variables = (*self._q_networks[key].variables,)

            target_variables = (*self._target_q_networks[key].variables,)

            # Make online -> target network update ops.
            if tf.math.mod(self._num_steps, self._target_update_period) == 0:
                for src, dest in zip(online_variables, target_variables):
                    dest.assign(src)
        self._num_steps.assign_add(1)

    def _forward(self, inputs: reverb.ReplaySample) -> None:
        """Trainer forward pass

        Args:
            inputs (Any): input data from the data table (transitions)
        """
        trans = mava_types.Transition(*inputs.data)

        # Unpack transition
        observations, next_observations, actions, rewards, discounts, _, _ = (
            trans.observation,
            trans.next_observation,
            trans.action,
            trans.reward,
            trans.discount,
            trans.extras,
            trans.next_extras,
        )

        # Dicts to store per network type
        (net_observation, 
        net_next_observation, 
        net_action, 
        net_reward, 
        net_discount) = self._get_feed_by_network_type(
            observations, 
            next_observations,
            actions,
            rewards,
            discounts
        )

        network_losses = {}
        with tf.GradientTape(persistent=True) as tape:
            
            for net_key in self.unique_net_keys:
                network_losses[net_key] = {}
                
                # Evaluate g-networks
                logits = self._g_networks[net_key](net_observation[net_key])
                cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                loss = cce(net_action[net_key], logits)

                loss = tf.reduce_mean(loss)
                network_losses[net_key]["g_network_loss"] = loss

                # Evaluate q-networks
                q_tm1 = self._q_networks[net_key](net_observation[net_key])
                q_t_value = self._target_q_networks[net_key](net_next_observation[net_key])
                q_t_selector = self._filtered_q_value(net_key, net_next_observation[net_key])

                # Compute the loss.
                loss, extra = trfl.double_qlearning(q_tm1, net_action[net_key], 
                        net_reward[net_key], net_discount[net_key], q_t_value, q_t_selector)

                # loss = losses.huber(extra.td_error, self._huber_loss_parameter)
                loss = tf.reduce_mean(loss, axis=[0])  # []
                
                network_losses[net_key]["q_network_loss"] = loss

        # Maybe update target network.
        self._update_target_networks()

        # Store losses and tape
        self._network_losses = network_losses
        self._tape = tape

    def _backward(self) -> None:
        """Trainer backward pass updating network parameters"""

        network_losses = self._network_losses
        tape = self._tape
        for net_key in self.unique_net_keys:

            # Get trainable variables
            q_network_variables = self._q_networks[net_key].trainable_variables
            g_network_variables = self._g_networks[net_key].trainable_variables

            # Compute gradients
            q_gradients = tape.gradient(
                network_losses[net_key]["q_network_loss"], q_network_variables)
            g_gradients = tape.gradient(
                network_losses[net_key]["g_network_loss"], g_network_variables)

            # Clip gradients.
            q_gradients = tf.clip_by_global_norm(q_gradients, self._max_gradient_norm)[0]
            g_gradients = tf.clip_by_global_norm(g_gradients, self._max_gradient_norm)[0]

            # Apply gradients.
            self._q_optimizers[net_key].apply(q_gradients, q_network_variables)
            self._g_optimizers[net_key].apply(g_gradients, g_network_variables)

        # Delete tape.
        train_utils.safe_del(self, "tape")

    # @tf.function
    def _step(self) -> Dict:
        """Trainer forward and backward passes."""

        # Get data from the offline dataset
        inputs = next(self._iterator)

        # Do Q-learning.
        self._forward(inputs)

        # Compute gradients.
        self._backward()

        # Set fetches to Q-value losses.
        fetches = self._network_losses

        # Return fetches.
        return fetches

    def step(self) -> None:
        """Trainer step to update the parameters of the system."""

        # Run the learning step.
        fetches = self._step()

        # Compute elapsed time.
        timestamp = time.time()
        if self._timestamp:
            elapsed_time = timestamp - self._timestamp
        else:
            elapsed_time = 0
        self._timestamp = timestamp  # type: ignore

        # Update our counts and record it.
        counts = self._counter.increment(steps=1, walltime=elapsed_time)
        fetches.update(counts)

        # Checkpoint and attempt to write the logs.
        if self._checkpoint:
            train_utils.checkpoint_networks(self._system_checkpointer)

        if self._logger:
            self._logger.write(fetches)

    def get_variables(self, names: Sequence[str]) -> Dict[str, Dict[str, np.ndarray]]:
        """Get network variables.

        Args:
            names: network names

        Returns:
            Dict[str, Dict[str, np.ndarray]]: network variables
        """
        variables: Dict[str, Dict[str, np.ndarray]] = {}
        for network_type in names:
            variables[network_type] = {
                agent: tf2_utils.to_numpy(
                    self._system_network_variables[network_type][agent]
                )
                for agent in self.unique_net_keys
            }
        return variables
