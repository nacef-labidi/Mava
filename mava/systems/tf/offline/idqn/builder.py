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

"""Offline IDQN system builder implementation."""

import dataclasses
from typing import Any, Dict, Iterator, Optional, Type, Union

import reverb
import sonnet as snt
from acme.tf import variable_utils
from acme.utils import counting

from mava import core, specs, types
from mava.systems.tf.offline.idqn import execution, training
from mava.wrappers.system_trainer_statistics import DetailedTrainerStatistics
from mava.components.tf.modules.exploration.exploration_scheduling import (
    BaseExplorationScheduler,
)

@dataclasses.dataclass
class OfflineIDQNConfig:
    environment_spec: specs.MAEnvironmentSpec
    agent_net_keys: Dict[str, str]
    discount: float
    batch_size: int
    target_update_period: int
    executor_variable_update_period: int
    max_gradient_norm: Optional[float]
    learning_rate: float
    checkpoint: bool
    checkpoint_subpath: str
    checkpoint_minute_interval: int
    evaluator_exploration_scheduler_fn: Type[BaseExplorationScheduler]
    evaluator_exploration_scheduler_kwargs: Dict
    distributional: bool


class OfflineIDQNBuilder:
    """Builder for offline IDQN which constructs components of the system."""

    def __init__(
        self,
        config: OfflineIDQNConfig,
        trainer_fn: Type[training.OfflineIDQNTrainer] = training.OfflineIDQNTrainer,
        executor_fn: Type[core.Executor] = execution.OfflineIDQNFeedForwardExecutor,
        extra_specs: Dict[str, Any] = {},
    ):
        """Initialise the system.

        Args:
            config: system configuration specifying hyperparameters and
                additional information for constructing the system.
            trainer_fn: Trainer function, of a
                correpsonding type to work with the selected system architecture.
                Defaults to training.OfflineMADQNTrainer.
            executor_fn: Executor function, of a
                corresponding type to work with the selected system architecture.
                Defaults to execution.MADQNFeedForwardExecutor.
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
        networks: Dict[str, Dict[str, snt.Module]],
        variable_source: Optional[core.VariableSource] = None,
    ) -> core.Executor:
        """Create an executor instance.

        Args:
            q_networks (Dict[str, snt.Module]): q-value networks for each agent in the
                system.
            action_selectors (Dict[str, Any]): policy action selector method, e.g.
                epsilon greedy.
            adder (Optional[adders.ParallelAdder], optional): adder to send data to
                a replay buffer. Defaults to None.
            variable_source (Optional[core.VariableSource], optional): variables server.

        Returns:
            core.Executor: system executor, a collection of agents making up the part
                of the system generating data by interacting the environment.
        """
        agent_net_keys = self._config.agent_net_keys

        # Q-networks
        q_networks = networks["q-networks"]
        action_selectors = networks["action_selectors"]

        # Network supports
        if self._config.distributional:
            network_supports = networks["supports"]

        # Variable updator
        variable_client = None
        if variable_source:
            # Get variables
            variables = {
                net_key: q_networks[net_key].variables
                for net_key in set(agent_net_keys.values())
            }
            # Get new network variables
            variable_client = variable_utils.VariableClient(
                client=variable_source,
                variables={"q_network": variables},
                update_period=self._config.executor_variable_update_period,
            )

            # Make sure not to use a random policy after checkpoint restoration by
            # assigning variables before running the environment loop.
            variable_client.update_and_wait()

        # Make exploration scheduler
        exploration_scheduler = self._config.evaluator_exploration_scheduler_fn(
            **self._config.evaluator_exploration_scheduler_kwargs
        )

        # Create the executor which coordinates the actors.
        return self._executor_fn(
            q_networks=q_networks,
            action_selectors=action_selectors,
            agent_net_keys=agent_net_keys,
            variable_client=variable_client,
            exploration_scheduler=exploration_scheduler,
            network_supports=network_supports,
            distributional=self._config.distributional
        )

    def make_trainer(
        self,
        networks: Dict[str, Dict[str, snt.Module]],
        dataset: reverb.ReplaySample,
        counter: Optional[counting.Counter] = None,
        logger: Optional[types.NestedLogger] = None,
    ) -> core.Trainer:
        """Create a trainer instance.

        Args:
            networks (Dict[str, Dict[str, snt.Module]]): system networks.
            dataset (Iterator[reverb.ReplaySample]): dataset iterator to feed data to
                the trainer networks.
            counter (Optional[counting.Counter], optional): a Counter which allows for
                recording of counts, e.g. trainer steps. Defaults to None.
            logger (Optional[types.NestedLogger], optional): Logger object for logging
                metadata.. Defaults to None.

        Returns:
            core.Trainer: system trainer, that uses the collected data from the
                executors to update the parameters of the agent networks in the system.
        """
        # Get Q-networks
        q_networks = networks["q-networks"]

        # Get network supports
        if self._config.distributional:
            network_supports = networks["supports"]
        else: network_supports = None

        # Get agent info
        agents = self._config.environment_spec.get_agent_ids()

        # The trainer updates the parameters
        trainer = self._trainer_fn(
            agents=agents,
            agent_net_keys=self._config.agent_net_keys,
            q_networks=q_networks,
            discount=self._config.discount,
            learning_rate=self._config.learning_rate,
            target_update_period=self._config.target_update_period,
            max_gradient_norm=self._config.max_gradient_norm,
            dataset=dataset,
            counter=counter,
            logger=logger,
            checkpoint=self._config.checkpoint,
            checkpoint_subpath=self._config.checkpoint_subpath,
            checkpoint_minute_interval=self._config.checkpoint_minute_interval,
            network_supports=network_supports,
            distributional=self._config.distributional
        )

        trainer = DetailedTrainerStatistics(
            trainer, metrics=["q_value_loss"]
        )  # type:ignore

        return trainer
