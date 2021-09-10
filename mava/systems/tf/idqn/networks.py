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
from acme.tf import utils as tf2_utils

from mava import specs as mava_specs
from mava.components.tf import networks
from mava.components.tf.networks import epsilon_greedy_action_selector

def make_default_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    agent_net_keys: Dict[str, str],
    q_network_layer_sizes: Sequence = (256,256),
) -> Mapping[str, types.TensorTransformation]:
    # Get env spec  
    specs = environment_spec.get_agent_specs()
    # Create agent_net specs
    specs = {agent_net_keys[key]: specs[key] for key in specs.keys()}

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
    for key, spec in specs.items():

        # Get total number of action dimensions from action spec.
        num_dimensions = specs[key].actions.num_values
        q_network = networks.LayerNormMLP(
            list(q_network_layer_sizes) + [num_dimensions],
            activate_final=False,
        )

        # Get observation spec for the network
        obs_spec = spec.observations.observation
        # Create variables for network.
        tf2_utils.create_variables(q_network, [obs_spec])

        # Store in dict
        q_networks[key] = q_network
        action_selectors[key] = action_selector_fn

    return {
        "q-networks": q_networks,
        "action_selectors": action_selectors,
    }
