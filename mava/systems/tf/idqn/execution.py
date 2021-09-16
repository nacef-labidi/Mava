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
"""IDQN system executor implementation."""
from typing import Dict, Optional, Union

import dm_env
import numpy as np
import sonnet as snt
import tensorflow as tf
from acme import types
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils
from acme.tf.networks.distributions import DiscreteValuedDistribution

from mava import adders
from mava.components.tf.modules import exploration
from mava.systems.tf.executors import FeedForwardExecutor
from mava.systems.tf.idqn.training import IDQNTrainer
from mava.types import OLT
from mava.components.tf.modules.exploration import BaseExplorationScheduler


class IDQNFeedForwardExecutor(FeedForwardExecutor):
    """A feed-forward executor.

    An executor based on a feed-forward policy for each agent in the system.
    """

    def __init__(
        self,
        q_networks: Dict[str, snt.Module],
        action_selectors: Dict[str, snt.Module],
        agent_net_keys: Dict[str, str],
        adder: Optional[adders.ParallelAdder] = None,
        variable_client: Optional[tf2_variable_utils.VariableClient] = None,
        distributional: bool = False,
        network_supports: Dict = None
    ):
        """Initialise the system executor

        Args:
            q_networks (Dict[str, snt.Module]): q-value networks for each agent in the
                system.
            action_selectors (Dict[str, Any]): policy action selector method, e.g.
                epsilon greedy.
            agent_net_keys: specifies what network each agent uses.
                Defaults to {}.
            adder: adder which sends data
                to a replay buffer. Defaults to None.
            variable_client: client to copy weights from the trainer.
                Defaults to None.
            evaluator: whether the executor will be used for
                evaluation. Defaults to False.
        """

        # Store these for later use.
        self._adder = adder
        self._variable_client = variable_client
        self._q_networks = q_networks
        self._action_selectors = action_selectors
        self._agent_net_keys = agent_net_keys

        # Distributional Q-learning stuff
        self._distributional = distributional
        self._network_supports = network_supports

    def _get_epsilon(self) -> Union[float, np.ndarray]:
        """Return epsilon.

        Returns:
            epsilon values.
        """
        data = list(
            {
                action_selector.get_epsilon()
                for action_selector in self._action_selectors.values()
            }
        )
        if len(data) == 1:
            return data[0]
        else:
            return np.array(list(data))

    def _decrement_epsilon(self) -> None:
        """Decrements epsilon in action selectors."""
        {
            action_selector.decrement_epsilon()
            for action_selector in self._action_selectors.values()
        }

    def get_stats(self) -> Dict:
        """Return extra stats to log.
        Returns:
            epsilon information.
        """
        return {
            f"{net_key}_epsilon": action_selector.get_epsilon()
            for net_key, action_selector in self._action_selectors.items()
        }

    def _policy(
        self,
        agent: str,
        observation: types.NestedTensor,
        legal_actions: types.NestedTensor,
    ) -> types.NestedTensor:
        """Agent specific policy function

        Args:
            agent (str): agent id
            observation (types.NestedTensor): observation tensor received from the
                environment.
            legal_actions (types.NestedTensor): actions allowed to be taken at the
                current observation.

        Returns:
            types.NestedTensor: agent action
        """

        # Add a dummy batch dimension and as a side effect convert numpy to TF.
        batched_observation = tf2_utils.add_batch_dim(observation)
        batched_legals = tf2_utils.add_batch_dim(legal_actions)

        # Get network ID
        net_key = self._agent_net_keys[agent]

        # Compute the policy, conditioned on the observation.
        if self._distributional:
            # Get network support
            support = self._network_supports[net_key]

            logits = self._q_networks[net_key](batched_observation)
            z = tf.nn.softmax(logits, axis=-1)
            q_values = tf.reduce_sum(z * support, axis=-1)   
        else:
            q_values = self._q_networks[net_key](batched_observation)

        # Select legal action
        action = self._action_selectors[net_key](
            q_values, batched_legals
        )

        return action

    @tf.function
    def do_policies(self, observations: types.NestedArray):
        actions = {}
        for agent, observation in observations.items():

            actions[agent] = self._policy(
                agent,
                observation.observation,
                observation.legal_actions
            )

        return actions

    def select_actions(
        self, observations: Dict[str, OLT]
    ) -> Dict[str, types.NestedArray]:
        """Select the actions for all agents in the system

        Args:
            observations (Dict[str, OLT]): transition object containing observations,
                legal actions and terminals.

        Returns:
            Dict[str, types.NestedArray]: actions for all agents in the system.
        """
        # Apply polisies
        actions = self.do_policies(observations)

        # Return a numpy arrays with squeezed out batch dimension.
        actions = tf2_utils.to_numpy_squeeze(actions)

        # Decrement epsilon
        self._decrement_epsilon()

        return actions

    def observe_first(
        self,
        timestep: dm_env.TimeStep,
        extras: Dict[str, types.NestedArray] = {},
    ) -> None:
        """Record first observed timestep from the environment

        Args:
            timestep (dm_env.TimeStep): data emitted by an environment at first step of
                interaction.
            extras (Dict[str, types.NestedArray], optional): possible extra information
                to record during the first step. Defaults to {}.
        """
        if self._adder:
            self._adder.add_first(timestep, extras)

    def observe(
        self,
        actions: Dict[str, types.NestedArray],
        next_timestep: dm_env.TimeStep,
        next_extras: Dict[str, types.NestedArray] = {},
    ) -> None:
        """Record observed timestep from the environment

        Args:
            actions (Dict[str, types.NestedArray]): system agents' actions.
            next_timestep (dm_env.TimeStep): data emitted by an environment during
                interaction.
            next_extras (Dict[str, types.NestedArray], optional): possible extra
                information to record during the transition. Defaults to {}.
        """
        if self._adder:
            self._adder.add(actions, next_timestep, next_extras)

    def update(self, wait: bool = False) -> None:
        """Update executor variables

        Args:
            wait: whether to stall the executor's request for new
                variables. Defaults to False.
        """
        if self._variable_client:
            self._variable_client.update(wait)