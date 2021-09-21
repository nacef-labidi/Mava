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
"""BCQ system executor implementation."""

from mava.components.tf.modules.exploration.exploration_scheduling import ConstantExplorationScheduler
from typing import Dict, Optional

import dm_env
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
from acme import types
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils

from mava.systems.tf.executors import FeedForwardExecutor
from mava.types import OLT
from mava.components.tf.modules.exploration import ConstantExplorationScheduler
from mava.components.tf.networks.epsilon_greedy import EpsilonGreedy


class BCQFeedForwardExecutor(FeedForwardExecutor):
    """A feed-forward executor.

    An executor based on a feed-forward policy for each agent in the system.
    """

    def __init__(
        self,
        q_networks: Dict[str, snt.Module],
        g_networks: Dict[str, snt.Module],
        agent_net_keys: Dict[str, str],
        threshold: float,
        variable_client: Optional[tf2_variable_utils.VariableClient] = None,
        distributional: bool = False,
        network_supports: Dict = None,
    ):
        """Initialise the system executor

        Args:
            q_networks: q-value networks for each agent in the system.
            g_networks: behavior cloning networks for each agent in the system.
            agent_net_keys: (dict, optional): specifies what network each agent uses.
                Defaults to {}.
            variable_client: client to copy weights from the trainer. Defaults to None.
        """

        # Store these for later use.
        self._variable_client = variable_client
        self._q_networks = q_networks
        self._g_networks = g_networks
        self._agent_net_keys = agent_net_keys

        # Greedy action selector
        self._action_selector = EpsilonGreedy(
            ConstantExplorationScheduler(
                epsilon_start=0.0,
                epsilon_decay=None,
                epsilon_min=None
            )
        )

        assert threshold >= 0 and threshold <= 1
        self._threshold = threshold

        # Distributional Q-learning stuff
        self._distributional = distributional
        self._network_supports = network_supports

    def _policy(
        self,
        agent: str,
        observation: types.NestedTensor,
        legal_actions: types.NestedTensor,
    ) -> types.NestedTensor:
        """Agent specific policy function

        Args:
            agent: agent id
            observation: observation tensor received from the
                environment.

        Returns:
            types.NestedTensor: agent action
        """

        # Add a dummy batch dimension and as a side effect convert numpy to TF.
        batched_observation = tf2_utils.add_batch_dim(observation)
        # TODO remove this when we add legals to offline dataset
        if legal_actions is not None:
            legal_actions = tf2_utils.add_batch_dim(legal_actions)

        # index network either on agent type or on agent id
        net_key = self._agent_net_keys[agent]

        # Compute q-values
        if self._distributional:
            # Get network support
            support = self._network_supports[net_key]

            logits = self._q_networks[net_key](batched_observation)
            z = tf.nn.softmax(logits, axis=-1)
            q_t = tf.reduce_sum(z * support, axis=-1)
        else:
            q_t = self._q_networks[net_key](batched_observation)

        # Compute behaviour cloning
        g_t = tf.nn.softmax(self._g_networks[net_key](batched_observation))
        normalized_g_t = g_t / tf.reduce_max(g_t, axis=-1, keepdims=True)

        # Filter actions based on g_network outputs.
        min_q = tf.reduce_min(q_t, axis=-1, keepdims=True)
        filtered_q = tf.where(normalized_g_t >= self._threshold, q_t, min_q)
        # TODO use action selector when we store legal_actions in dataset
        # action = self._action_selector(filtered_q, legallegal_actions)
        action = tf.argmax(filtered_q, axis=1)

        return action

    @tf.function
    def do_policies(self, observations: types.NestedArray):
        actions = {}
        for agent, observation in observations.items():

            actions[agent] = self._policy(
                agent,
                observation.observation,
                observation.legal_actions,
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
