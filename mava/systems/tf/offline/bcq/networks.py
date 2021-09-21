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
from typing import Dict, Mapping, Sequence, Union

import sonnet as snt
import tensorflow as tf
from acme import types
from acme.tf import utils as tf2_utils
import acme.tf.networks as acme_networks

from mava import specs as mava_specs
from mava.components.tf import networks


def make_default_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    agent_net_keys: Dict[str, str],
    network_layer_sizes: Union[Dict[str, Sequence], Sequence] = None,
    distributional: bool = False,
    num_atoms: int = 51,
    vmin: int = -100,
    vmax: int = 100,
) -> Mapping[str, types.TensorTransformation]:
    """Default networks for BCQ.

    Args:
        environment_spec: description of the action and
            observation spaces etc. for each agent in the system.
        agent_net_keys: specifies what network each agent uses.
            Defaults to {}.
        network_layer_sizes: size of networks.

    Returns:
        Mapping[str, types.TensorTransformation]: returned agent networks.
    """
    # Get agent specs.
    specs = environment_spec.get_agent_specs()

    # Create agent_type specs
    specs = {agent_net_keys[key]: specs[key] for key in specs.keys()}

    # Set layer sizes.
    if not network_layer_sizes:
        network_layer_sizes = (256, 256)

    all_networks = {"q-networks": {}, "g-networks": {}, "supports": {}}
    for key, spec in specs.items():

        # Get total number of action dimensions from action spec.
        num_dimensions = specs[key].actions.num_values

        # If distributional
        if distributional:
            q_network = snt.Sequential(
                [
                    networks.LayerNormMLP(
                        list(network_layer_sizes) + [num_atoms * num_dimensions],
                        activate_final=False,
                    ),
                    tf.keras.layers.Reshape((num_dimensions, num_atoms)),
                ]
            )
            network_support = tf.cast(
                tf.linspace(vmin, vmax, num_atoms), dtype="float32"
            )
        else:
            # Create the networks.
            q_network = acme_networks.LayerNormMLP(
                list(network_layer_sizes) + [num_dimensions],
                activate_final=False,
            )

        g_network = acme_networks.LayerNormMLP(
            list(network_layer_sizes) + [num_dimensions],
            activate_final=False,
        )

        # Get observation spec for policy.
        obs_spec = spec.observations.observation
        # Create variables for value and policy networks.
        tf2_utils.create_variables(q_network, [obs_spec])
        tf2_utils.create_variables(g_network, [obs_spec])

        # Store network and action selector.
        all_networks["q-networks"][key] = q_network
        all_networks["g-networks"][key] = g_network
        all_networks["supports"][key] = network_support

    return all_networks
