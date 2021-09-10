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
"""Behaviour cloning system trainer implementation."""

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


class BCTrainer(mava.Trainer):
    """Behaviour Cloning trainer.

    This is the trainer component of an BC system. IE it takes a dataset
    as input and implements update functionality to learn from this dataset.
    """

    def __init__(
        self,
        agents: List[str],
        agent_types: List[str],
        networks: Dict[str, snt.Module],
        dataset: tf.data.Dataset,
        optimizer: Union[Dict[str, snt.Optimizer], snt.Optimizer],
        agent_net_keys: Dict[str, str],
        max_gradient_norm: float = None,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        checkpoint: bool = True,
        checkpoint_subpath: str = "~/mava/",
        checkpoint_minute_interval: int = 15,
    ):
        """Initialise Offline MADQN trainer.

        Args:
            agents (List[str]): agent ids, e.g. "agent_0".
            agent_types (List[str]): agent types, e.g. "speaker" or "listener".
            networks (Dict[str, snt.Module]): networks being optimized.
            dataset (tf.data.Dataset): training dataset.
            optimizer (Union[snt.Optimizer, Dict[str, snt.Optimizer]]): type of
                optimizer for updating the parameters of the networks.
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

        # Store networks
        self._networks = networks

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

        # Create optimizers for different agent types.
        if not isinstance(optimizer, dict):
            self._optimizers: Dict[str, snt.Optimizer] = {}
            for agent in self.unique_net_keys:
                self._optimizers[agent] = copy.deepcopy(optimizer)
        else:
            self._optimizers = optimizer

        # Expose the variables.
        networks_to_expose = {}
        self._system_network_variables: Dict[str, Dict[str, snt.Module]] = {
            "network": {},
        }
        for agent_key in self.unique_net_keys:
            network_to_expose = self._networks[agent_key]

            networks_to_expose[agent_key] = network_to_expose

            self._system_network_variables["network"][
                agent_key
            ] = network_to_expose.variables

        # Checkpointer
        self._system_checkpointer = {}
        if checkpoint:
            for agent_key in self.unique_net_keys:

                checkpointer = tf2_savers.Checkpointer(
                    directory=checkpoint_subpath,
                    time_delta_minutes=checkpoint_minute_interval,
                    objects_to_save={
                        "counter": self._counter,
                        "network": self._networks[agent_key],
                        "optimizer": self._optimizers,
                        "num_steps": self._num_steps,
                    },
                    enable_checkpointing=checkpoint,
                )

                self._system_checkpointer[agent_key] = checkpointer

        # Do not record timestamps until after the first learning step is done.
        self._timestamp: Optional[float] = None

    def _forward(self, inputs: reverb.ReplaySample) -> None:
        """Trainer forward pass

        Args:
            inputs (Any): input data from the data table (transitions)
        """
        trans = mava_types.Transition(*inputs.data)

        observations, _, actions, _, _, _, _ = (
            trans.observation,
            trans.next_observation,
            trans.action,
            trans.reward,
            trans.discount,
            trans.extras,
            trans.next_extras,
        )

        net_observation = {}
        net_action = {}
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

        network_losses = {}
        with tf.GradientTape(persistent=True) as tape:
            
            for net_key in self.unique_net_keys:
                # Evaluate our networks
                logits = self._networks[net_key](net_observation[net_key])
                cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                loss = cce(net_action[net_key], logits)

                loss = tf.reduce_mean(loss)
                network_losses[net_key] = {"network_loss": loss}

        # Store losses and tape
        self._network_losses = network_losses
        self._tape = tape

    def _backward(self) -> None:
        """Trainer backward pass updating network parameters"""

        network_losses = self._network_losses
        tape = self._tape

        for net_key in self.unique_net_keys:

            # Get trainable variables
            network_variables = self._networks[net_key].trainable_variables

            # Compute gradients
            gradients = tape.gradient(network_losses[net_key], network_variables)

            # Clip gradients.
            gradients = tf.clip_by_global_norm(gradients, self._max_gradient_norm)[0]

            # Apply gradients.
            self._optimizers[net_key].apply(gradients, network_variables)

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
