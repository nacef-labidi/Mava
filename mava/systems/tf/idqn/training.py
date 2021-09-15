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

"""Independent DQN system trainer implementation."""
import os
import copy
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import reverb
import sonnet as snt
import tensorflow as tf
import trfl
from acme.tf import losses
from acme.tf import utils as tf2_utils
from acme.types import NestedArray
from acme.utils import counting, loggers
from acme.tf.networks.distributions import DiscreteValuedDistribution

import mava
from mava import types as mava_types
from mava.adders import reverb as reverb_adders
from mava.systems.tf import savers as tf2_savers
from mava.utils import training_utils as train_utils

train_utils.set_growing_gpu_memory()


class IDQNTrainer(mava.Trainer):
    """IDQN trainer.
    This is the trainer component of a IDQN system. IE it takes a dataset as input
    and implements update functionality to learn from this dataset.
    """

    def __init__(
        self,
        agents: List[str],
        agent_net_keys: Dict[str, str],
        q_networks: Dict[str, snt.Module],
        dataset: tf.data.Dataset,
        learning_rate: float = 1e-3,
        discount: float = 0.99,
        target_update_period: int = 100,
        max_gradient_norm: Optional[float] = None,
        counter: Optional[counting.Counter] = None,
        logger: Optional[loggers.Logger] = None,
        checkpoint: bool = True,
        checkpoint_minute_interval: int = 15,
        checkpoint_subpath: str = "~/mava/",
        distributional: bool = False,
        network_supports: Dict = None
    ):
        """Independet DQN Trainer.

        Args:
            agents (List[str]): [description]
            agent_net_keys (Dict[str, str]): [description]
            q_networks (Dict[str, snt.Module]): [description]
            dataset (tf.data.Dataset): [description]
            learning_rate (float, optional): [description]. Defaults to 1e-3.
            discount (float, optional): [description]. Defaults to 0.99.
            target_update_period (int, optional): [description]. Defaults to 100.
            importance_sampling_exponent (float, optional): [description]. Defaults to 0.2.
            max_gradient_norm (Optional[float], optional): [description]. Defaults to None.
            replay_client (Optional[reverb.TFClient], optional): [description]. Defaults to None.
            counter (Optional[counting.Counter], optional): [description]. Defaults to None.
            logger (Optional[loggers.Logger], optional): [description]. Defaults to None.
            checkpoint (bool, optional): [description]. Defaults to True.
            checkpoint_minute_interval (int, optional): [description]. Defaults to 15.
            checkpoint_subpath (str, optional): [description]. Defaults to "~/mava/".
        """
        # Agent information
        self._agents = agents
        self._agent_net_keys = agent_net_keys
        self._unique_net_keys = set(self._agent_net_keys.values())

        # Store online q-networks.
        self._q_networks = q_networks
        # Create target q-networks.
        self._target_q_networks: Dict[str, snt.Module] = {}
        for key, net in q_networks.items():
            self._target_q_networks[key] = copy.deepcopy(net)

        # Distributional Q-learning stuff
        self._distributional = distributional
        self._network_supports = network_supports

        # Store hyper-parameters
        self._discount = discount
        self._target_update_period = target_update_period

        # Create optimizers
        self._optimizers: Dict[str, snt.optimizers.Adam] = {}
        for net_key in agent_net_keys.values():
            self._optimizers[net_key] = snt.optimizers.Adam(learning_rate)

        # Set up gradient clipping.
        if max_gradient_norm is not None:
            self._max_gradient_norm = tf.convert_to_tensor(max_gradient_norm)
        else:  # A very large number. Infinity results in NaNs.
            self._max_gradient_norm = tf.convert_to_tensor(1e10)

        # General learner book-keeping and loggers.
        self._num_steps = tf.Variable(0, dtype=tf.int32)
        self._counter = counter or counting.Counter()
        self._logger = logger

        # Create an iterator to go through the dataset.
        self._iterator = iter(dataset)

        # Expose the network variables.
        q_networks_to_expose = {}
        self._system_network_variables: Dict[str, Dict[str, snt.Module]] = {
            "q_network": {},
        }
        for net_key in self._unique_net_keys:
            q_network_to_expose = self._target_q_networks[net_key]

            q_networks_to_expose[net_key] = q_network_to_expose

            self._system_network_variables["q_network"][
                net_key
            ] = q_network_to_expose.variables

        # Setup checkpointer
        self._checkpoint = checkpoint
        if self._checkpoint:
            self._system_checkpointer = {}
            for net_key in self._unique_net_keys:

                subdir = os.path.join("trainer", net_key)

                checkpointer = tf2_savers.Checkpointer(
                    directory=checkpoint_subpath,
                    time_delta_minutes=checkpoint_minute_interval,
                    objects_to_save={
                        "counter": self._counter,
                        "q_network": self._q_networks[net_key],
                        "target_q_network": self._target_q_networks[net_key],
                        "optimizer": self._optimizers[net_key],
                        "num_steps": self._num_steps,
                    },
                    subdirectory=subdir,
                    enable_checkpointing=checkpoint,
                )

                self._system_checkpointer[net_key] = checkpointer

        # Do not record timestamps until after the first learning step is done.
        # This is to avoid including the time it takes for actors to come online and
        # fill the replay buffer.
        self._timestamp: Optional[float] = None

    def _update_target_networks(self) -> None:
        """Sync the target network parameters with the latest online network
        parameters"""
        for net_key in self._unique_net_keys:
            online_variables = (*self._q_networks[net_key].variables,)
            target_variables = (*self._target_q_networks[net_key].variables,)

            # Make online -> target network update ops.
            if tf.math.mod(self._num_steps, self._target_update_period) == 0:
                for src, dest in zip(online_variables, target_variables):
                    dest.assign(src)
        self._num_steps.assign_add(1)

    def _get_feed_by_network_type(
        self, 
        observations, 
        next_observations, 
        actions, 
        rewards, 
        discounts
    ) -> Tuple:
        # Initalize dicts
        net_observations = {}
        net_next_observations = {}
        net_actions = {}
        net_rewards = {}
        net_discounts = {}
        for agent in self._agents:
            net_key = self._agent_net_keys[agent]

            # Concat observations
            if net_key not in net_observations.keys():
                net_observations[net_key] = observations[agent].observation
            else:
                net_observations[net_key] = tf.concat(
                    [
                        net_observations[net_key],
                        observations[agent].observation
                    ],
                    axis=0
                )

            # Concat next observations
            if net_key not in net_next_observations.keys():
                net_next_observations[net_key] = next_observations[agent].observation
            else:
                net_next_observations[net_key] = tf.concat(
                    [
                        net_next_observations[net_key],
                        next_observations[agent].observation
                    ],
                    axis=0
                )

            # Concat actions
            if net_key not in net_actions.keys():
                net_actions[net_key] = actions[agent]
            else:
                net_actions[net_key] = tf.concat(
                    [
                       net_actions[net_key],
                        actions[agent]
                    ],
                    axis=0
                )

            # Concat rewards
            if net_key not in net_rewards.keys():
                net_rewards[net_key] = rewards[agent]
            else:
                net_rewards[net_key] = tf.concat(
                    [
                       net_rewards[net_key],
                        rewards[agent]
                    ],
                    axis=0
                )

            # Concat discounts
            if net_key not in net_discounts.keys():
                net_discounts[net_key] = discounts[agent]
            else:
                net_discounts[net_key] = tf.concat(
                    [
                       net_discounts[net_key],
                        discounts[agent]
                    ],
                    axis=0
                )

        return (net_observations, net_next_observations, 
            net_actions, net_rewards, net_discounts)

    @tf.function
    def _step(self) -> Dict:
        """Trainer forward and backward passes."""

        # Get data from replay (dropping extras if any). Note there is no
        # extra data here because we do not insert any into Reverb.
        inputs = next(self._iterator)

        # Compute losses
        self._forward(inputs)

        # Apply gradients
        self._backward()

        # Log losses
        fetches = self._q_network_losses

        return fetches

    def _forward(self, inputs: reverb.ReplaySample) -> None:
        """Trainer forward pass

        Args:
            inputs: input data from the data table (transitions)
        """
        # Get transitions data
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

        # Get feed by network type
        (net_observations, net_next_observations, net_actions, 
            net_rewards, net_discounts) = self._get_feed_by_network_type(
                observations, next_observations, actions, rewards, discounts
            )

        self._q_network_losses: Dict[str, NestedArray] = {}
        with tf.GradientTape(persistent=True) as tape:

            for net_key in self._unique_net_keys:
                
                if self._distributional:
                    support = self._network_supports[net_key]

                    logits = self._q_networks[net_key](net_observations[net_key])
                    target_logits = self._target_q_networks[net_key](net_next_observations[net_key])

                    q_t_selector_logits = self._q_networks[net_key](net_next_observations[net_key])
                    q_t_selector_z = tf.nn.softmax(q_t_selector_logits, axis=-1)
                    q_t_selector = tf.reduce_sum(q_t_selector_z * support, axis=-1)  

                    # trfl distributional double Q-learning
                    loss, _ = trfl.categorical_dist_double_qlearning(
                        support,
                        logits,
                        net_actions[net_key],
                        net_rewards[net_key],
                        self._discount * net_discounts[net_key],
                        support, 
                        target_logits,
                        q_t_selector
                    )
                else:
                    q_tm1 = self._q_networks[net_key](net_observations[net_key])
                    q_t_value = self._target_q_networks[net_key](net_next_observations[net_key])
                    q_t_selector = self._q_networks[net_key](net_next_observations[net_key])

                    # trfl double Q-learning
                    loss, _ = trfl.double_qlearning(
                        q_tm1,
                        net_actions[net_key],
                        net_rewards[net_key],
                        self._discount * net_discounts[net_key],
                        q_t_value,
                        q_t_selector,
                    )

                # Store loss
                loss = tf.reduce_mean(loss)  # []
                self._q_network_losses[net_key] = {"q_value_loss": loss}

        # Store gradient tape
        self._tape = tape

    def _backward(self) -> None:
        """Trainer backward pass updating network parameters"""
        q_network_losses = self._q_network_losses
        tape = self._tape
        for net_key in self._unique_net_keys:
            # Get trainable variables
            q_network_variables = self._q_networks[net_key].trainable_variables

            # Compute gradients
            gradients = tape.gradient(q_network_losses[net_key], q_network_variables)

            # Clip gradients
            gradients = tf.clip_by_global_norm(gradients, self._max_gradient_norm)[0]

            # Apply gradients.
            self._optimizers[net_key].apply(gradients, q_network_variables)

        train_utils.safe_del(self, "tape")

        # Maybe update the target networks
        self._update_target_networks()

    def step(self) -> None:
        """Trainer step to update the parameters of the agents in the system"""

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
        """get network variables

        Args:
            names: network names

        Returns:
            Dict: network variables
        """

        variables: Dict[str, Dict[str, np.ndarray]] = {}
        for network_type in names:
            variables[network_type] = {
                agent: tf2_utils.to_numpy(
                    self._system_network_variables[network_type][agent]
                )
                for agent in self._unique_net_keys
            }
        return variables