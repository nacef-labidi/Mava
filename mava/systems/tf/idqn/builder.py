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

"""IDQN system builder implementation."""

import dataclasses
from typing import Any, Dict, Iterator, List, Optional, Type, Union

import reverb
import sonnet as snt
from acme import datasets
from acme.tf import variable_utils
from acme.utils import counting

from mava import adders, core, specs, types
from mava.adders import reverb as reverb_adders
from mava.components.tf.modules.exploration.exploration_scheduling import (
    BaseExplorationScheduler,
)
from mava.systems.tf.idqn import execution, training
from mava.wrappers.system_trainer_statistics import DetailedTrainerStatistics



@dataclasses.dataclass
class IDQNConfig:
    """Configuration options for the IDQN system.

    Args:
        environment_spec: description of the action and observation spaces etc. for
            each agent in the system.
        epsilon_min: final minimum value for epsilon at the end of a decay schedule.
        epsilon_decay: the rate at which epislon decays.
        agent_net_keys: specifies what network each agent uses.
            Defaults to {}.
        target_update_period: number of learner steps to perform before updating
            the target networks.
        executor_variable_update_period: the rate at which executors sync their
            paramters with the trainer.
        max_gradient_norm: value to specify the maximum clipping value for the gradient
            norm during optimization.
        min_replay_size: minimum replay size before updating.
        max_replay_size: maximum replay size.
        samples_per_insert: number of samples to take from replay for every insert
            that is made.
        prefetch_size: size to prefetch from replay.
        batch_size: batch size for updates.
        n_step: number of steps to include prior to boostrapping.
        discount: discount to use for TD updates.
        checkpoint: boolean to indicate whether to checkpoint models.
        checkpoint_minute_interval: The number of minutes to wait between
            checkpoints.
        checkpoint_subpath: subdirectory specifying where to store checkpoints.
    """

    environment_spec: specs.MAEnvironmentSpec
    agent_net_keys: Dict[str, str]
    executor_exploration_scheduler_fn: Type[BaseExplorationScheduler]
    evaluator_exploration_scheduler_fn: Type[BaseExplorationScheduler]
    executor_exploration_scheduler_kwargs: Dict
    evaluator_exploration_scheduler_kwargs: Dict
    target_update_period: int
    learning_rate: float
    executor_variable_update_period: int
    max_gradient_norm: Optional[float]
    min_replay_size: int
    max_replay_size: int
    samples_per_insert: Optional[float]
    prefetch_size: int
    batch_size: int
    n_step: int
    discount: float
    checkpoint: bool
    checkpoint_minute_interval: int
    checkpoint_subpath: str


class IDQNBuilder:
    """Builder for IDQN which constructs individual components of the system."""

    def __init__(
        self,
        config: IDQNConfig,
        trainer_fn: Type[training.IDQNTrainer] = training.IDQNTrainer,
        executor_fn: Type[core.Executor] = execution.IDQNFeedForwardExecutor,
        extra_specs: Dict[str, Any] = {},
    ):
        """Initialise the system.

        Args:
            config (MADQNConfig): system configuration specifying hyperparameters and
                additional information for constructing the system.
            trainer_fn: Trainer function, of a
                correpsonding type to work with the selected system architecture.
                Defaults to training.MADQNTrainer.
            executor_fn: Executor function, of a
                corresponding type to work with the selected system architecture.
                Defaults to execution.MADQNFeedForwardExecutor.
            extra_specs (Dict[str, Any], optional): defines the specifications of extra
                information used by the system. Defaults to {}.
        """
        # Config and extra spec
        self._config = config
        self._extra_specs = extra_specs
        # Agent info
        self._agents = self._config.environment_spec.get_agent_ids()
        self._agent_types = self._config.environment_spec.get_agent_types()
        # Trainer and executor
        self._trainer_fn = trainer_fn
        self._executor_fn = executor_fn

    def make_replay_tables(
        self,
        environment_spec: specs.MAEnvironmentSpec,
    ) -> List[reverb.Table]:
        """Create tables to insert data into.

        Args:
            environment_spec (specs.MAEnvironmentSpec): description of the action and
                observation spaces etc. for each agent in the system.

        Raises:
            NotImplementedError: unknown executor type.

        Returns:
            List[reverb.Table]: a list of data tables for inserting data.
        """

        # Select adder
        adder_sig = reverb_adders.ParallelNStepTransitionAdder.signature(
            environment_spec, self._extra_specs
        )

        # Setup rate limiter
        if self._config.samples_per_insert is None:
            # We will take a samples_per_insert ratio of None to mean that there is
            # no limit, i.e. this only implies a min size limit.
            limiter = reverb.rate_limiters.MinSize(self._config.min_replay_size)

        else:
            # Create enough of an error buffer to give a 10% tolerance in rate.
            samples_per_insert_tolerance = 0.1 * self._config.samples_per_insert
            error_buffer = self._config.min_replay_size * samples_per_insert_tolerance
            limiter = reverb.rate_limiters.SampleToInsertRatio(
                min_size_to_sample=self._config.min_replay_size,
                samples_per_insert=self._config.samples_per_insert,
                error_buffer=error_buffer,
            )

        # Create replay table
        replay_table = reverb.Table(
            name=reverb_adders.DEFAULT_PRIORITY_TABLE,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=self._config.max_replay_size,
            rate_limiter=limiter,
            signature=adder_sig,
        )

        return [replay_table]

    def make_dataset(
        self, replay_client: reverb.Client
    ) -> Iterator[reverb.ReplaySample]:
        """Create a dataset to use for training/updating the system.

        Args:
            replay_client (reverb.Client): Reverb Client which points to the
                replay server.

        Returns:
            [type]: dataset iterator.

        Yields:
            Iterator[reverb.ReplaySample]: data samples from the dataset.
        """
        # Create dataset
        dataset = datasets.make_reverb_dataset(
            table=reverb_adders.DEFAULT_PRIORITY_TABLE,
            server_address=replay_client.server_address,
            batch_size=self._config.batch_size,
            prefetch_size=self._config.prefetch_size,
        )

        return dataset

    def make_adder(
        self, replay_client: reverb.Client
    ) -> Optional[adders.ParallelAdder]:
        """Create an adder which records data generated by the executor/environment.

        Args:
            replay_client: Reverb Client which points to the replay server.

        Raises:
            NotImplementedError: unknown executor type.

        Returns:
            Optional[adders.ParallelAdder]: adder which sends data to a replay buffer.
        """

        # Select adder
        adder = reverb_adders.ParallelNStepTransitionAdder(
            client=replay_client,
            n_step=self._config.n_step,
            discount=self._config.discount,
        )

        return adder

    def make_executor(
        self,
        q_networks: Dict[str, snt.Module],
        action_selectors: Dict[str, Any],
        adder: Optional[adders.ParallelAdder] = None,
        variable_source: Optional[core.VariableSource] = None,
        is_evaluator: bool = False
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
        if is_evaluator:
            exploration_scheduler = self._config.evaluator_exploration_scheduler_fn(
                **self._config.evaluator_exploration_scheduler_kwargs
            )
        else:
            exploration_scheduler = self._config.executor_exploration_scheduler_fn(
                **self._config.executor_exploration_scheduler_kwargs
            )

        # Create the executor which coordinates the actors.
        return self._executor_fn(
            q_networks=q_networks,
            action_selectors=action_selectors,
            agent_net_keys=agent_net_keys,
            variable_client=variable_client,
            adder=adder,
            exploration_scheduler=exploration_scheduler
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
        )

        trainer = DetailedTrainerStatistics(
            trainer, metrics=["q_value_loss"]
        )  # type:ignore

        return trainer
