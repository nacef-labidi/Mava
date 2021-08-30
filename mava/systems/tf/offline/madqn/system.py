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
"""Offline MADQN system implementation."""

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
from mava.components.tf.architectures import DecentralisedValueActor
from mava.environment_loop import ParallelEnvironmentLoop
from mava.systems.tf import executors
from mava.systems.tf import savers as tf2_savers
from mava.systems.tf.offline.madqn import builder, execution, training
from mava.utils import lp_utils
from mava.utils.loggers import MavaLogger, logger_utils
from mava.wrappers import DetailedPerAgentStatistics


class OfflineMADQN:
    """Offline MADQN system."""

    def __init__(
        self,
        dataset_factory: Callable[[], tf.data.Dataset],
        environment_factory: Callable[[bool], dm_env.Environment],
        network_factory: Callable[[acme_specs.BoundedArray], Dict[str, snt.Module]],
        logger_factory: Callable[[str], MavaLogger] = None,
        architecture: Type[DecentralisedValueActor] = DecentralisedValueActor,
        trainer_fn: Type[training.OfflineMADQNTrainer] = training.OfflineMADQNTrainer,
        executor_fn: Type[core.Executor] = execution.OfflineMADQNFeedForwardExecutor,
        environment_spec: mava_specs.MAEnvironmentSpec = None,
        shared_weights: bool = True,
        agent_net_keys: Dict[str, str] = {},
        batch_size: int = 256,
        max_gradient_norm: float = None,
        discount: float = 0.99,
        optimizer: Union[snt.Optimizer, Dict[str, snt.Optimizer]] = snt.optimizers.Adam(
            learning_rate=1e-4
        ),
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
    ):
        """Initialise an offline MADQN system.

        Args:
            dataset_factory: function to create an offline dataset.
            environment_factory: function to create environment for evaluation.
            network_factory: function to create network.
            logger_factory: function to create logger. Defaults to None.
            architecture: system architecture. Defaults to DecentralisedValueActor.
            trainer_fn: system trainer function.
                Defaults to training.OfflineMADQNTrainer.
            executor_fn: system executor function. Defaults to
                execution.OfflineMADQNFeedForwardExecutor.
            environment_spec: environment spec. Defaults to None.
            shared_weights: agents with same type should share network weights.
                Defaults to True.
            agent_net_keys: agent network keys. Defaults to {}.
            batch_size: training batch size. Defaults to 256.
            max_gradient_norm: max gradient norm, used for gradient
                clipping. Defaults to None.
            discount: extra agent discount. Defaults to 0.99.
            optimizer: network optimizers. Defaults to
                snt.optimizers.Adam( learning_rate=1e-4 ).
            target_update_period: number of trainer steps before
                target update. Defaults to 100.
            executor_variable_update_period: trainer update steps
                before executors get new variables. Defaults to 1000.
            max_executor_steps: max number of executor steps before
                termination. Defaults to None.
            checkpoint: should the system checkpoint. Defaults to True.
            checkpoint_subpath: path to checkpoint dir. Defaults to "~/mava/".
            checkpoint_minute_interval: time between checkpoints. Defaults to 15.
            logger_config: extra configs for logger. Defaults to {}.
            train_loop_fn: trainer loop. Defaults to ParallelEnvironmentLoop.
            eval_loop_fn: eval loop. Defaults to ParallelEnvironmentLoop.
            train_loop_fn_kwargs: train loop kwargs. Defaults to {}.
            eval_loop_fn_kwargs: eval loop kwargs. Defaults to {}.
        """
        # Store the dataset factory.
        self._dataset_factory = dataset_factory

        if not environment_spec:
            environment_spec = mava_specs.MAEnvironmentSpec(
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

        # Store useful variables for later
        self._architecture = architecture
        self._environment_factory = environment_factory
        self._network_factory = network_factory
        self._logger_factory = logger_factory
        self._environment_spec = environment_spec
        self._max_executor_steps = max_executor_steps
        self._checkpoint_subpath = checkpoint_subpath
        self._checkpoint = checkpoint
        self._logger_config = logger_config
        self._train_loop_fn = train_loop_fn
        self._train_loop_fn_kwargs = train_loop_fn_kwargs
        self._eval_loop_fn = eval_loop_fn
        self._eval_loop_fn_kwargs = eval_loop_fn_kwargs

        # Setup agent networks
        self._agent_net_keys = agent_net_keys
        if not agent_net_keys:
            agents = environment_spec.get_agent_ids()
            self._agent_net_keys = {
                agent: agent.split("_")[0] if shared_weights else agent
                for agent in agents
            }
        assert not issubclass(executor_fn, executors.RecurrentExecutor)

        # Make the builder
        self._builder = builder.OfflineMADQNBuilder(
            builder.OfflineMADQNConfig(
                environment_spec=environment_spec,
                agent_net_keys=self._agent_net_keys,
                discount=discount,
                batch_size=batch_size,
                target_update_period=target_update_period,
                executor_variable_update_period=executor_variable_update_period,
                max_gradient_norm=max_gradient_norm,
                checkpoint=checkpoint,
                optimizer=optimizer,
                checkpoint_subpath=checkpoint_subpath,
                checkpoint_minute_interval=checkpoint_minute_interval,
            ),
            trainer_fn=trainer_fn,
            executor_fn=executor_fn,
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

        # Create the networks to optimize
        networks = self._network_factory(  # type: ignore
            environment_spec=self._environment_spec,
            agent_net_keys=self._agent_net_keys,
        )

        # Create system architecture with target networks.
        architecture = self._architecture(
            environment_spec=self._environment_spec,
            value_networks=networks["q_networks"],
            agent_net_keys=self._agent_net_keys,
        )

        system_networks = architecture.create_system()

        # Create logger
        trainer_logger_config = {}
        if self._logger_config and "trainer" in self._logger_config:
            trainer_logger_config = self._logger_config["trainer"]
        trainer_logger = self._logger_factory(  # type: ignore
            "trainer", **trainer_logger_config
        )

        dataset = iter(self._dataset_factory())
        counter = counting.Counter(counter, "trainer")

        return self._builder.make_trainer(
            networks=system_networks,
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

        # Create the behavior policy.
        networks = self._network_factory(  # type: ignore
            environment_spec=self._environment_spec,
            agent_net_keys=self._agent_net_keys,
        )

        # Create system architecture with target networks.
        architecture = self._architecture(
            environment_spec=self._environment_spec,
            value_networks=networks["q_networks"],
            agent_net_keys=self._agent_net_keys,
        )

        system_networks = architecture.create_system()

        # Create the agent.
        executor = self._builder.make_executor(
            q_networks=system_networks["values"],
            action_selectors=networks["action_selectors"],
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

        # Agent statistics wrapper
        eval_loop = DetailedPerAgentStatistics(eval_loop)

        return eval_loop

    def build(self, name: str = "madqn") -> Any:
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
            program.add_node(lp.CourierNode(self.evaluator, trainer, counter, trainer))

        return program
