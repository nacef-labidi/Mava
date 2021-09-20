import functools
import os
from datetime import datetime
from typing import Any, Dict

import launchpad as lp
from absl import app, flags
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator

from mava.adders.tfrecord import TFRecordParallelTransitionAdder
from mava.components.tf.modules.exploration import ConstantExplorationScheduler
from mava.systems.tf import idqn
from mava.utils import lp_utils
from mava.utils.environments.flatland_utils import flatland_env_factory
from mava.utils.loggers import logger_utils

mava_id = str(datetime.now())
base_dir = "logs"
log_dir = os.path.join(base_dir, mava_id)

# Checkpointer appends "checkpoints" to checkpoint_dir
checkpoint_dir = "logs/online_idqn"


def main(_: Any) -> None:
    """Main function for the example.

    Args:
        _ (Any): args.
    """

    # Flatland environment config
    rail_gen_cfg: Dict = {
        "max_num_cities": 3,
        "max_rails_between_cities": 2,
        "max_rails_in_city": 3,
        "grid_mode": False,
    }

    flatland_env_config: Dict = {
        "number_of_agents": 5,
        "width": 25,
        "height": 25,
        "rail_generator": sparse_rail_generator(**rail_gen_cfg),
        "schedule_generator": sparse_schedule_generator(),
        "obs_builder_object": TreeObsForRailEnv(
            max_depth=2, predictor=ShortestPathPredictorForRailEnv(max_depth=30)
        ),
    }

    # Environment factory
    environment_factory = functools.partial(
        flatland_env_factory, env_config=flatland_env_config, include_agent_info=False
    )

    # Networks factory
    network_factory = lp_utils.partial_kwargs(
        idqn.make_default_networks, q_network_layer_sizes=(128,)
    )

    # Log every [log_every] seconds.
    log_every = 10
    logger_factory = functools.partial(
        logger_utils.make_logger,
        directory=base_dir,
        to_terminal=True,
        to_tensorboard=True,
        time_stamp=mava_id,
        time_delta=log_every,
    )

    # Build distributed program
    program = idqn.IDQNEvaluator(
        checkpoint_subpath=checkpoint_dir,
        log_subpath=log_dir,
        environment_factory=environment_factory,
        network_factory=network_factory,
        logger_factory=logger_factory,
        evaluator_exploration_scheduler_fn=ConstantExplorationScheduler,
        evaluator_exploration_scheduler_kwargs={
            "epsilon_start": 0.001,
            "epsilon_min": None,
            "epsilon_decay": None,
        },
        tfrecord_adder_factory=TFRecordParallelTransitionAdder,
        tfrecord_adder_kwargs={"transitions_per_file": 100_000},
    ).build()

    # Ensure only trainer runs on gpu, while other processes run on cpu.
    local_resources = lp_utils.to_device(
        program_nodes=program.groups.keys(), nodes_on_gpu=[]
    )

    # Launch.
    lp.launch(
        program,
        lp.LaunchType.LOCAL_MULTI_PROCESSING,
        terminal="current_terminal",
        local_resources=local_resources,
    )


if __name__ == "__main__":
    app.run(main)
