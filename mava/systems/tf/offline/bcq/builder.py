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

"""Batch Constrained Q-learning system builder implementation."""

import dataclasses
from typing import Any, Dict, Iterator, Optional, Type, Union

import reverb
import sonnet as snt
from acme.tf import variable_utils
from acme.utils import counting

from mava import core, specs, types
from mava.systems.tf.offline.bcq import execution, training
from mava.wrappers.system_trainer_statistics import DetailedTrainerStatistics


@dataclasses.dataclass
class BCQConfig:
    environment_spec: specs.MAEnvironmentSpec
    agent_net_keys: Dict[str, str]
    batch_size: int
    discount: float
    learning_rate: float
    target_update_period: int
    huber_loss_parameter: float
    threshold: float
    executor_variable_update_period: int
    max_gradient_norm: Optional[float]
    checkpoint: bool
    checkpoint_subpath: str
    checkpoint_minute_interval: int


class BCQBuilder:
    """Builder for BCQ which constructs components of the system."""

    def __init__(
        self,
        config: BCQConfig,
        trainer_fn: Type[training.BCQTrainer] = training.BCQTrainer,
        executor_fn: Type[core.Executor] = execution.BCQFeedForwardExecutor,
        extra_specs: Dict[str, Any] = {},
    ):
        """Initialise the system.

        Args:
            config: system configuration specifying hyperparameters and
                additional information for constructing the system.
            trainer_fn: Trainer function, of a
                correpsonding type to work with the selected system architecture.
            executor_fn: Executor function, of a
                corresponding type to work with the selected system architecture.
            extra_specs: defines the specifications of extra
                information used by the system. Defaults to {}.
        """
        self._config = config
        self._extra_specs = extra_specs
        self._trainer_fn = trainer_fn
        self._executor_fn = executor_fn
        self._agents = self._config.environment_spec.get_agent_ids()
        self._agent_types = self._config.environment_spec.get_agent_types()

    def make_executor(
        self,
        q_networks: Dict[str, snt.Module],
        g_networks: Dict[str, snt.Module],
        variable_source: Optional[core.VariableSource] = None,
    ) -> core.Executor:
        """Create an executor instance.

        Args:
            q_networks: q-networks in the system.
            g_networks: g-networks in the system.
            variable_source: variables server. Defaults to None.

        Returns:
            core.Executor: system executor, a collection of agents making up the part
                of the system generating data by interacting the environment.
        """
        # Get agent net keys
        agent_net_keys = self._config.agent_net_keys

        variable_client = None
        if variable_source:
            # Create variables
            q_variables = {
                net_key: q_networks[net_key].variables
                for net_key in set(agent_net_keys.values())
            }
            g_variables = {
                net_key: g_networks[net_key].variables
                for net_key in set(agent_net_keys.values())
            }
            # Get new variables
            variable_client = variable_utils.VariableClient(
                client=variable_source,
                variables={
                    "q_network": q_variables,
                    "g_network": g_variables
                },
                update_period=self._config.executor_variable_update_period,
            )

            # Make sure not to use a random policy after checkpoint restoration by
            # assigning variables before running the environment loop.
            variable_client.update_and_wait()

        # Create the executor which coordinates the actors.
        return self._executor_fn(
            q_networks=q_networks,
            g_networks=g_networks,
            threshold=self._config.threshold,
            agent_net_keys=agent_net_keys,
            variable_client=variable_client,
        )

    def make_trainer(
        self,
        q_networks: Dict[str, snt.Module],
        g_networks: Dict[str, snt.Module],
        dataset: Iterator[reverb.ReplaySample],
        counter: Optional[counting.Counter] = None,
        logger: Optional[types.NestedLogger] = None,
    ) -> core.Trainer:
        """Create a trainer instance.

        Args:
            q_networks: q-networks to optimize.
            q_networks: g-networks to optimize.
            dataset: dataset iterator to feed data to the trainer networks.
            counter: a Counter which allows for recording of counts, 
                e.g. trainer steps. Defaults to None.
            logger: Logger object for logging metadata.. Defaults to None.

        Returns:
            core.Trainer: system trainer, that uses the collected data from the
                executors to update the parameters of the agent networks in the system.
        """
        agents = self._config.environment_spec.get_agent_ids()
        agent_types = self._config.environment_spec.get_agent_types()

        # The learner updates the parameters (and initializes them).
        trainer = self._trainer_fn(
            agents=agents,
            agent_types=agent_types,
            q_networks=q_networks,
            g_networks=g_networks,
            discount=self._config.discount,
            agent_net_keys=self._config.agent_net_keys,
            learning_rate=self._config.learning_rate,
            huber_loss_parameter=self._config.huber_loss_parameter,
            target_update_period=self._config.target_update_period,
            threshold=self._config.threshold,
            max_gradient_norm=self._config.max_gradient_norm,
            dataset=dataset,
            counter=counter,
            logger=logger,
            checkpoint=self._config.checkpoint,
            checkpoint_subpath=self._config.checkpoint_subpath,
            checkpoint_minute_interval=self._config.checkpoint_minute_interval,
        )

        trainer = DetailedTrainerStatistics(
            trainer, metrics=["q_network_loss", "g_network_loss"]
        )  # type:ignore

        return trainer
