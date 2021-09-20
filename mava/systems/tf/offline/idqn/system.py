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
"""Offline IDQN system implementation."""

import functools
from typing import Any, Callable, Dict, Type, Union

import acme
import dm_env
import launchpad as lp
import sonnet as snt
import tensorflow as tf
from acme import specs as acme_specs
from acme.utils import counting

import mava
from mava import core
from mava import specs as mava_specs
from mava.components.tf.modules.exploration import (
    BaseExplorationScheduler,
    ConstantExplorationScheduler,
)
from mava.environment_loop import ParallelEnvironmentLoop
from mava.systems.tf import executors
from mava.systems.tf import savers as tf2_savers
from mava.systems.tf.offline.idqn import builder, execution, training
from mava.utils import lp_utils
from mava.utils.loggers import MavaLogger, logger_utils
from mava.wrappers import DetailedPerAgentStatistics


class OfflineIDQN:
    """Offline IDQN system."""

    def __init__(
        self,
        dataset_factory: Callable[[], tf.data.Dataset],
        environment_factory: Callable[[bool], dm_env.Environment],
        network_factory: Callable[[acme_specs.BoundedArray], Dict[str, snt.Module]],
        logger_factory: Callable[[str], MavaLogger] = None,
        trainer_fn: Type[training.OfflineIDQNTrainer] = training.OfflineIDQNTrainer,
        executor_fn: Type[core.Executor] = execution.OfflineIDQNFeedForwardExecutor,
        environment_spec: mava_specs.MAEnvironmentSpec = None,
        shared_weights: bool = True,
        agent_net_keys: Dict[str, str] = {},
        batch_size: int = 256,
        max_gradient_norm: float = None,
        discount: float = 0.99,
        learning_rate: float = 1e-4,
        target_update_period: int = 100,
        executor_variable_update_period: int = 1000,
        max_executor_steps: int = None,
        checkpoint: bool = True,
        checkpoint_subpath: str = "~/mava/",
        checkpoint_minute_interval: int = 15,
        logger_config: Dict = {},
        train_loop_fn: Callable = ParallelEnvironmentLoop,
        eval_loop_fn: Callable = ParallelEnvironmentLoop,
        train_loop_fn_kwargs: Dict = {},
        eval_loop_fn_kwargs: Dict = {},
        evaluator_exploration_scheduler_fn: Type[
            BaseExplorationScheduler
        ] = ConstantExplorationScheduler,
        evaluator_exploration_scheduler_kwargs: Dict = {
            "epsilon_start": 0.0,
            "epsilon_min": None,
            "epsilon_decay": None,
        },
        distributional: bool = False,
    ):
        # Make environment spec
        self._environment_spec = mava_specs.MAEnvironmentSpec(
            environment_factory(evaluation=False)  # type:ignore
        )

        # Set default logger if no logger provided
        if not logger_factory:
            logger_factory = functools.partial(
                logger_utils.make_logger,
                directory="~/mava",
                to_terminal=True,
                time_delta=10,
            )

        # Store factories
        self._environment_factory = environment_factory
        self._network_factory = network_factory
        self._logger_factory = logger_factory
        self._dataset_factory = dataset_factory

        # Setup agent networks
        self._agent_net_keys = agent_net_keys
        if not agent_net_keys:
            agents = self._environment_spec.get_agent_ids()
            self._agent_net_keys = {
                agent: agent.split("_")[0] if shared_weights else agent
                for agent in agents
            }

        self._max_executor_steps = max_executor_steps
        self._logger_config = logger_config
        self._train_loop_fn = train_loop_fn
        self._train_loop_fn_kwargs = train_loop_fn_kwargs
        self._eval_loop_fn = eval_loop_fn
        self._eval_loop_fn_kwargs = eval_loop_fn_kwargs
        self._checkpoint_minute_interval = checkpoint_minute_interval
        self._checkpoint_subpath = checkpoint_subpath
        self._checkpoint = checkpoint

        # Empty Extras spec
        extra_specs = {}

        self._builder = builder.OfflineIDQNBuilder(
            builder.OfflineIDQNConfig(
                environment_spec=self._environment_spec,
                agent_net_keys=self._agent_net_keys,
                evaluator_exploration_scheduler_fn=evaluator_exploration_scheduler_fn,
                evaluator_exploration_scheduler_kwargs=evaluator_exploration_scheduler_kwargs,
                discount=discount,
                batch_size=batch_size,
                learning_rate=learning_rate,
                target_update_period=target_update_period,
                executor_variable_update_period=executor_variable_update_period,
                max_gradient_norm=max_gradient_norm,
                checkpoint=checkpoint,
                checkpoint_subpath=checkpoint_subpath,
                checkpoint_minute_interval=checkpoint_minute_interval,
                distributional=distributional,
            ),
            trainer_fn=trainer_fn,
            executor_fn=executor_fn,
            extra_specs=extra_specs,
        )

    def counter(self, checkpoint: bool) -> Any:
        """Step counter

        Args:
            checkpoint (bool): whether to checkpoint the counter.

        Returns:
            Any: checkpointing object logging steps in a counter subdirectory.
        """

        if checkpoint:
            return tf2_savers.CheckpointingRunner(
                counting.Counter(),
                time_delta_minutes=self._builder._config.checkpoint_minute_interval,
                directory=self._checkpoint_subpath,
                subdirectory="counter",
            )
        else:
            return counting.Counter()

    def coordinator(self, counter: counting.Counter) -> Any:
        """Coordination helper for a distributed program

        Args:
            counter (counting.Counter): step counter object.

        Returns:
            Any: step limiter object.
        """

        return lp_utils.StepsLimiter(counter, self._max_executor_steps)  # type: ignore

    def trainer(
        self,
        counter: counting.Counter,
    ) -> mava.core.Trainer:
        """System trainer

        Args:
            counter (counting.Counter): step counter object.

        Returns:
            mava.core.Trainer: system trainer.
        """
        # Make the dataset
        dataset = iter(self._dataset_factory().batch(self._builder._config.batch_size))

        # Create the networks to optimize (online)
        networks = self._network_factory(  # type: ignore
            environment_spec=self._environment_spec,
            agent_net_keys=self._agent_net_keys,
        )

        # Create logger
        trainer_logger_config = {}
        if self._logger_config and "trainer" in self._logger_config:
            trainer_logger_config = self._logger_config["trainer"]
        trainer_logger = self._logger_factory(  # type: ignore
            "trainer", **trainer_logger_config
        )

        # Make counter
        counter = counting.Counter(counter, "trainer")

        return self._builder.make_trainer(
            networks=networks,
            dataset=dataset,
            counter=counter,
            logger=trainer_logger,
        )

    def evaluator(
        self,
        variable_source: acme.VariableSource,
        counter: counting.Counter,
    ) -> Any:
        """System evaluator i.e. an executor process not connected to a dataset.

        Args:
            variable_source: variable server for updating network variables.
            counter: step counter object.

        Returns:
            Any: environment-executor evaluation loop instance for evaluating the
                performance of a system.
        """
        # Create the networks
        networks = self._network_factory(  # type: ignore
            environment_spec=self._environment_spec,
            agent_net_keys=self._agent_net_keys,
        )

        # Create the agent.
        executor = self._builder.make_executor(
            networks=networks,
            variable_source=variable_source,
        )

        # Make the environment.
        environment = self._environment_factory(evaluation=True)  # type: ignore

        # Create logger and counter.
        counter = counting.Counter(counter, "evaluator")
        evaluator_logger_config = {}
        if self._logger_config and "evaluator" in self._logger_config:
            evaluator_logger_config = self._logger_config["evaluator"]
        eval_logger = self._logger_factory(  # type: ignore
            "evaluator", **evaluator_logger_config
        )

        # Create the run loop and return it.
        # Create the loop to connect environment and executor.
        eval_loop = self._eval_loop_fn(
            environment,
            executor,
            counter=counter,
            logger=eval_logger,
            **self._eval_loop_fn_kwargs,
        )

        # Stats wrapper
        eval_loop = DetailedPerAgentStatistics(eval_loop)

        return eval_loop

    def build(self, name: str = "offline-idqn") -> Any:
        """Build the distributed system as a graph program.

        Args:
            name (str, optional): system name. Defaults to "madqn".

        Returns:
            Any: graph program for distributed system training.
        """

        program = lp.Program(name=name)

        with program.group("counter"):
            counter = program.add_node(lp.CourierNode(self.counter, self._checkpoint))

        if self._max_executor_steps:
            with program.group("coordinator"):
                _ = program.add_node(lp.CourierNode(self.coordinator, counter))

        with program.group("trainer"):
            trainer = program.add_node(lp.CourierNode(self.trainer, counter))

        with program.group("evaluator"):
            program.add_node(lp.CourierNode(self.evaluator, trainer, counter))

        return program
