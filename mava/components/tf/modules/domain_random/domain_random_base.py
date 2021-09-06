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

import copy
import time
from typing import Any, Dict, List, Union

import numpy as np
import tensorflow as tf

from mava.systems.tf.mad4pg.execution import MAD4PGRecurrentExecutor
from mava.systems.tf.mad4pg.system import MAD4PG
from mava.systems.tf.maddpg.builder import MADDPGBuilder
from mava.systems.tf.maddpg.execution import MADDPGRecurrentExecutor
from mava.systems.tf.maddpg.system import MADDPG
from mava.systems.tf.variable_sources import VariableSource as MavaVariableSource
from mava.utils.sort_utils import sample_new_agent_keys, sort_str_num

"""Domain random training interface for multi-agent RL systems."""
supported_dr_systems = [MADDPG, MAD4PG]
supported_dr_executors = [MADDPGRecurrentExecutor, MAD4PGRecurrentExecutor]


# DR variable source
class DRVariableSource(MavaVariableSource):
    def __init__(
        self,
        variables: Dict[str, Any],
        checkpoint: bool,
        checkpoint_subpath: str,
        unique_net_keys: List[str],
    ) -> None:
        """Initialise the variable source
        Args:
            variables (Dict[str, Any]): a dictionary with
            variables which should be stored in it.
            checkpoint (bool): Indicates whether checkpointing should be performed.
            checkpoint_subpath (str): checkpoint path
        Returns:
            None
        """
        # TODO: Fix these values
        self._diff_inc_score_threshold = 0.1
        self._checkpoint_interval = 10 * 60

        self._last_checkpoint_time = time.time()
        super().__init__(
            variables=variables,
            checkpoint=checkpoint,
            checkpoint_subpath=checkpoint_subpath,
        )

    def run(self) -> None:
        """Run the variable source. This function allows for
        checkpointing and other centralised computations to
        be performed by the variable source.
                Args:
                    None
                Returns:
                    None
        """
        # Checkpoints every 5 minutes
        while True:
            # Wait 10 seconds before checking again
            time.sleep(10)

            # Check if system should checkpoint (+1 second to make sure the
            # checkpointer does the save)
            if (
                self._system_checkpointer
                and self._last_checkpoint_time + self._checkpoint_interval + 1
                < time.time()
            ):
                self._last_checkpoint_time = time.time()
                self._system_checkpointer.save()
                tf.print("Saved latest variables.")

            # Update the best response networks

            avg_executor_score = self.variables["moving_avg_rewards"]
            import tensorflow as tf

            print("avg_executor_score: ", avg_executor_score)
            tf.print("avg_executor_score: ", avg_executor_score)
            if avg_executor_score > self._diff_inc_score_threshold:
                # Set the moving average to 0.9 to allow for adjusting period
                # not to update game_diff.
                self.variables["moving_avg_rewards"].assign(
                    self._diff_inc_score_threshold * 0.9
                )
                # Only increase the environmental difficulty.
                self.variables["game_diff"].assign(
                    np.clip(self.variables["game_diff"] + 0.01, 0.0, 1.0)
                )
                # print("game_diff: ", self.variables["game_diff"])
                # tf.print("game_diff: ", self.variables["game_diff"])


def DomainRandomWrapper(  # noqa
    system: Union[MADDPG, MAD4PG],  # noqa
) -> Union[MADDPG, MAD4PG]:
    """Initializes the droadcaster communicator.
    Args:
        system: The system that should be wrapped.
    Returns:
        system: The wrapped system.
    """
    if type(system) not in supported_dr_systems:
        raise NotImplementedError(
            f"Currently only the {supported_dr_systems} systems have "
            f"the correct hooks to support this wrapper. Not {type(system)}."
        )

    if system._builder._executor_fn not in supported_dr_executors:
        raise NotImplementedError(
            f"Currently only the {supported_dr_executors} executors have "
            f"the correct hooks to support this wrapper. "
            f"Not {system._builder._executor_fn}."
        )

    # Wrap the executor with the DR hooks
    class DRExecutor(system._builder._executor_fn):  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)

    system._builder._executor_fn = DRExecutor

    # Wrap the executor with the DR hooks
    class DRExecutor(system._builder._executor_fn):  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            # Connect the environment difficulty to the variable_client difficulty.
            self._environment.game_diff = self._variable_client._variables["game_diff"]

        # DR executor
        def _custom_end_of_episode_logic(self) -> None:
            """Custom logic at the end of an episode."""
            mvg_avg_weighting = 0.01
            for agent in self._cum_rewards.keys():
                # TODO (dries): Is this correct?
                # If a network is used more in training its moving average will be
                # updated faster. Should we use the average reward of the networks?
                self._variable_client.move_avg_and_wait(
                    f"moving_avg_rewards",
                    self._cum_rewards[agent],
                    mvg_avg_weighting,
                )
                self._cum_rewards[agent] = 0.0

    system._builder._executor_fn = DRExecutor

    # Wrap the system builder with the DR hooks
    class DRBuilder(type(system._builder)):  # type: ignore
        def __init__(
            self,
            builder: MADDPGBuilder,
        ):
            """Initialise the system.
            Args:
                builder: The builder to wrap.
            """

            self.__dict__ = builder.__dict__

        def create_custom_executor_variables(
            self,
            variables: Dict[str, tf.Variable],
            get_keys: List[str] = None,
        ) -> None:
            """Create counter variables.
            Args:
                variables (Dict[str, snt.Variable]): dictionary with variable_source
                variables in.
                get_keys (List[str]): list of keys to get from the variable server.
            Returns:
                None.
            """
            vars = {}
            vars[f"game_diff"] = tf.Variable(0.0, dtype=tf.float32)
            vars[f"moving_avg_rewards"] = tf.Variable(0.0, dtype=tf.float32)
            if get_keys:
                get_keys.extend(list(vars.keys()))
            variables.update(vars)

        def variable_server_fn(
            self,
            *args: Any,
            **kwargs: Any,
        ) -> DRVariableSource:
            """Create a variable server.
            Args:
                *args: Variable arguments.
                **kwargs: Variable keyword arguments.
            Returns:
                a DRVariableSource.
            """
            return DRVariableSource(*args, **kwargs)

    system._builder = DRBuilder(system._builder)
    return system
