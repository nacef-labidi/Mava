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
"""Offline MADQN system trainer implementation."""

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
from acme.utils import counting, loggers

import mava
from mava import types as mava_types
from mava.systems.tf import savers as tf2_savers
from mava.utils import training_utils as train_utils

train_utils.set_growing_gpu_memory()


class OfflineMADQNTrainer(mava.Trainer):
    """Offline MADQN trainer.

    This is the trainer component of an offline MADQN system. IE it takes a dataset
    as input and implements update functionality to learn from this dataset.
    """

    def __init__(
        self,
        agents: List[str],
        agent_types: List[str],
        q_networks: Dict[str, snt.Module],
        target_q_networks: Dict[str, snt.Module],
        target_update_period: int,
        dataset: tf.data.Dataset,
        optimizer: Union[Dict[str, snt.Optimizer], snt.Optimizer],
        discount: float,
        agent_net_keys: Dict[str, str],
        max_gradient_norm: float = None,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        checkpoint: bool = True,
        checkpoint_subpath: str = "~/mava/",
        checkpoint_minute_interval: int = 15,
    ):
        """Initialise MADQN trainer.

        Args:
            agents (List[str]): agent ids, e.g. "agent_0".
            agent_types (List[str]): agent types, e.g. "speaker" or "listener".
            q_networks (Dict[str, snt.Module]): q-value networks.
            target_q_networks (Dict[str, snt.Module]): target q-value networks.
            target_update_period (int): number of steps before updating target networks.
            dataset (tf.data.Dataset): training dataset.
            optimizer (Union[snt.Optimizer, Dict[str, snt.Optimizer]]): type of
                optimizer for updating the parameters of the networks.
            discount (float): discount factor for TD updates.
            agent_net_keys: (dict, optional): specifies what network each agent uses.
                Defaults to {}.
            max_gradient_norm (float, optional): maximum allowed norm for gradients
                before clipping is applied. Defaults to None.
            counter (counting.Counter, optional): step counter object. Defaults to None.
            logger (loggers.Logger, optional): logger object for logging trainer
                statistics. Defaults to None.
            checkpoint (bool, optional): whether to checkpoint networks. Defaults to
                True.
            checkpoint_subpath (str, optional): subdirectory for storing checkpoints.
                Defaults to "~/mava/".
            checkpoint_minute_interval:  time in minutes between checkpoints.
        """
        self._agents = agents
        self._agent_types = agent_types
        self._agent_net_keys = agent_net_keys
        self._checkpoint = checkpoint

        # Store online and target q-networks.
        self._q_networks = q_networks
        self._target_q_networks = target_q_networks

        # General learner book-keeping and loggers.
        self._counter = counter or counting.Counter()
        self._logger = logger

        # Other learner parameters.
        self._discount = discount

        # Set up gradient clipping.
        if max_gradient_norm is not None:
            self._max_gradient_norm = tf.convert_to_tensor(max_gradient_norm)
        else:  # A very large number. Infinity results in NaNs.
            self._max_gradient_norm = tf.convert_to_tensor(1e10)

        # Necessary to track when to update target networks.
        self._num_steps = tf.Variable(0, dtype=tf.int32)
        self._target_update_period = target_update_period

        # Create an iterator to go through the dataset.
        self._iterator = dataset

        # Dictionary with network keys for each agent.
        self.unique_net_keys = set(self._agent_net_keys.values())

        # Create optimizers for different agent types.
        if not isinstance(optimizer, dict):
            self._optimizers: Dict[str, snt.Optimizer] = {}
            for agent in self.unique_net_keys:
                self._optimizers[agent] = copy.deepcopy(optimizer)
        else:
            self._optimizers = optimizer

        # Expose the variables.
        q_networks_to_expose = {}
        self._system_network_variables: Dict[str, Dict[str, snt.Module]] = {
            "q_network": {},
        }
        for agent_key in self.unique_net_keys:
            q_network_to_expose = self._target_q_networks[agent_key]

            q_networks_to_expose[agent_key] = q_network_to_expose

            self._system_network_variables["q_network"][
                agent_key
            ] = q_network_to_expose.variables

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
                        "target_q_network": self._target_q_networks[agent_key],
                        "optimizer": self._optimizers,
                        "num_steps": self._num_steps,
                    },
                    enable_checkpointing=checkpoint,
                )

                self._system_checkpointer[agent_key] = checkpointer

        # Do not record timestamps until after the first learning step is done.
        self._timestamp: Optional[float] = None

    def _update_target_networks(self) -> None:
        """Sync the target network parameters with latest online parameters"""

        for key in self.unique_net_keys:
            # Update target network.
            online_variables = (*self._q_networks[key].variables,)

            target_variables = (*self._target_q_networks[key].variables,)

            # Make online -> target network update ops.
            if tf.math.mod(self._num_steps, self._target_update_period) == 0:
                for src, dest in zip(online_variables, target_variables):
                    dest.assign(src)
        self._num_steps.assign_add(1)

    def _get_feed(
        self,
        o_tm1_trans: Dict[str, np.ndarray],
        o_t_trans: Dict[str, np.ndarray],
        a_tm1: Dict[str, np.ndarray],
        agent: str,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Get data to feed to the agent networks

        Args:
            o_tm1_trans: transformed (e.g. using observation
                network) observation at timestep t-1
            o_t_trans: transformed observation at timestep t
            a_tm1: action at timestep t-1
            agent: agent id

        Returns:
            Tuple: agent network feeds, observations
                at t-1, t and action at time t.
        """
        # Decentralised
        o_tm1_feed = o_tm1_trans[agent].observation
        o_t_feed = o_t_trans[agent].observation
        a_tm1_feed = a_tm1[agent]

        return o_tm1_feed, o_t_feed, a_tm1_feed

    def _forward(self, inputs: reverb.ReplaySample) -> None:
        """Trainer forward pass

        Args:
            inputs (Any): input data from the data table (transitions)
        """
        trans = mava_types.Transition(*inputs.data)

        o_tm1, o_t, a_tm1, r_t, d_t, _, _ = (
            trans.observation,
            trans.next_observation,
            trans.action,
            trans.reward,
            trans.discount,
            trans.extras,
            trans.next_extras,
        )

        with tf.GradientTape(persistent=True) as tape:
            q_network_losses: Dict[str, NestedArray] = {}

            for agent in self._agents:
                agent_key = self._agent_net_keys[agent]

                # Cast the additional discount to match the environment discount dtype.
                discount = tf.cast(self._discount, dtype=d_t[agent].dtype)

                # Maybe transform the observation before feeding into policy and critic.
                # Transforming the observations this way at the start of the learning
                # step effectively means that the policy and critic share observation
                # network weights.

                o_tm1_feed, o_t_feed, a_tm1_feed = self._get_feed(
                    o_tm1, o_t, a_tm1, agent
                )

                # Double Q-learning.
                q_tm1 = self._q_networks[agent_key](o_tm1_feed)
                q_t_value = self._target_q_networks[agent_key](o_t_feed)
                q_t_selector = self._q_networks[agent_key](o_t_feed)

                # Q-network learning
                loss, loss_extras = trfl.double_qlearning(
                    q_tm1,
                    a_tm1_feed,
                    r_t[agent],
                    discount * d_t[agent],
                    q_t_value,
                    q_t_selector,
                )

                loss = tf.reduce_mean(loss)
                q_network_losses[agent] = {"q_value_loss": loss}

        # Store losses and tape
        self._q_network_losses = q_network_losses
        self.tape = tape

    def _backward(self) -> None:
        """Trainer backward pass updating network parameters"""

        q_network_losses = self._q_network_losses
        tape = self.tape
        for agent in self._agents:
            agent_key = self._agent_net_keys[agent]

            # Get trainable variables
            q_network_variables = self._q_networks[agent_key].trainable_variables

            # Compute gradients
            gradients = tape.gradient(q_network_losses[agent], q_network_variables)

            # Clip gradients.
            gradients = tf.clip_by_global_norm(gradients, self._max_gradient_norm)[0]

            # Apply gradients.
            self._optimizers[agent_key].apply(gradients, q_network_variables)

        # Delete tape.
        train_utils.safe_del(self, "tape")

    @tf.function
    def _step(self) -> Dict:
        """Trainer forward and backward passes."""

        # Maybe update the target networks
        self._update_target_networks()

        # Get data from the offline dataset
        inputs = next(self._iterator)

        # Do Q-learning.
        self._forward(inputs)

        # Compute gradients.
        self._backward()

        # Set fetches to Q-value losses.
        fetches = self._q_network_losses

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
