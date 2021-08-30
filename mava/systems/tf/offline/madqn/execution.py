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
"""MADQN system executor implementation."""

from typing import Dict, Optional

import dm_env
import sonnet as snt
import tensorflow as tf
from acme import types
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils

from mava.systems.tf.executors import FeedForwardExecutor
from mava.types import OLT


class OfflineMADQNFeedForwardExecutor(FeedForwardExecutor):
    """A feed-forward executor.

    An executor based on a feed-forward policy for each agent in the system.
    """

    def __init__(
        self,
        q_networks: Dict[str, snt.Module],
        action_selectors: Dict[str, snt.Module],
        agent_net_keys: Dict[str, str],
        variable_client: Optional[tf2_variable_utils.VariableClient] = None,
    ):
        """Initialise the system executor

        Args:
            q_networks (Dict[str, snt.Module]): q-value networks for each agent in the
                system.
            action_selectors (Dict[str, Any]): policy action selector method, e.g.
                epsilon greedy.
            agent_net_keys: (dict, optional): specifies what network each agent uses.
                Defaults to {}.
            variable_client: client to copy weights from the trainer. Defaults to None.
        """

        # Store these for later use.
        self._variable_client = variable_client
        self._q_networks = q_networks
        self._action_selectors = action_selectors
        self._agent_net_keys = agent_net_keys

    @tf.function
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

        # index network either on agent type or on agent id
        agent_net_key = self._agent_net_keys[agent]

        # Compute the policy, conditioned on the observation
        q_values = self._q_networks[agent_net_key](batched_observation)

        # Select legal action
        action = self._action_selectors[agent_net_key](q_values, batched_legals)

        return action

    def select_action(
        self, agent: str, observation: types.NestedArray
    ) -> types.NestedArray:
        """Select an action for a single agent in the system

        Args:
            agent (str): agent id
            observation (types.NestedArray): observation tensor received from the
                environment.

        Returns:
            types.NestedArray: agent action
        """

        action = self._policy(
            agent,
            observation.observation,
            observation.legal_actions,
        )

        # Squeeze out batch dim and convert to numpy
        action = tf2_utils.to_numpy_squeeze(action)

        return action

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

        actions = {}
        for agent, observation in observations.items():

            actions[agent] = self.select_action(agent, observation)

        # Return a numpy array with squeezed out batch dimension.
        return actions

    def observe_first(
        self,
        timestep: dm_env.TimeStep,
        extras: Dict[str, types.NestedArray] = {},
    ) -> None:
        """Record first observed timestep from the environment.

        This is a no-op for the offline executor.

        Args:
            timestep (dm_env.TimeStep): data emitted by an environment at first step of
                interaction.
            extras (Dict[str, types.NestedArray], optional): possible extra information
                to record during the first step. Defaults to {}.
        """
        pass

    def observe(
        self,
        actions: Dict[str, types.NestedArray],
        next_timestep: dm_env.TimeStep,
        next_extras: Dict[str, types.NestedArray] = {},
    ) -> None:
        """Record observed timestep from the environment.

        This is a no-op for the offline executor.

        Args:
            actions (Dict[str, types.NestedArray]): system agents' actions.
            next_timestep (dm_env.TimeStep): data emitted by an environment during
                interaction.
            next_extras (Dict[str, types.NestedArray], optional): possible extra
                information to record during the transition. Defaults to {}.
        """
        pass

    def update(self, wait: bool = False) -> None:
        """Update executor variables

        Args:
            wait (bool, optional): whether to stall the executor's request for new
                variables. Defaults to False.
        """

        if self._variable_client:
            self._variable_client.update(wait)
