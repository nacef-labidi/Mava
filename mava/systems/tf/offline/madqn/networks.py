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
from typing import Dict, Mapping, Optional, Sequence, Union

import tensorflow as tf
from acme import types
from acme.tf import networks

from mava import specs as mava_specs
from mava.components.tf.networks import epsilon_greedy_action_selector


def make_default_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    agent_net_keys: Dict[str, str],
    networks_layer_sizes: Union[Dict[str, Sequence], Sequence] = None,
) -> Mapping[str, types.TensorTransformation]:
    """Default networks for offline MADQN.

    Args:
        environment_spec: description of the action and
            observation spaces etc. for each agent in the system.
        agent_net_keys: specifies what network each agent uses.
            Defaults to {}.
        networks_layer_sizes: size of networks.

    Returns:
        Mapping[str, types.TensorTransformation]: returned agent networks.
    """
    # Get agent specs.
    specs = environment_spec.get_agent_specs()

    # Create agent_type specs
    specs = {agent_net_keys[key]: specs[key] for key in specs.keys()}

    # Set layer sizes.
    if not networks_layer_sizes:
        networks_layer_sizes = (256, 256)

    if isinstance(networks_layer_sizes, Sequence):
        networks_layer_sizes = {key: networks_layer_sizes for key in specs.keys()}

    # Greedy action selector.
    def action_selector_fn(
        q_values: types.NestedTensor,
        legal_actions: types.NestedTensor,
        epsilon: Optional[tf.Variable] = None,
    ) -> types.NestedTensor:
        return epsilon_greedy_action_selector(
            action_values=q_values, legal_actions_mask=legal_actions, epsilon=epsilon
        )

    q_networks = {}
    action_selectors = {}
    for key in specs.keys():

        # Get total number of action dimensions from action spec.
        num_dimensions = specs[key].actions.num_values

        # Create the networks.
        q_network = networks.LayerNormMLP(
            list(networks_layer_sizes[key]) + [num_dimensions],
            activate_final=False,
        )

        # Store network and action selector.
        q_networks[key] = q_network
        action_selectors[key] = action_selector_fn

    return {
        "q_networks": q_networks,
        "action_selectors": action_selectors,
    }
