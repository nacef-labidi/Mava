import functools
from datetime import datetime
from typing import Any, Dict

import launchpad as lp
import sonnet as snt
from absl import app, flags
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator

from mava.specs import MAEnvironmentSpec
from mava.systems.tf.offline import bc
from mava.utils import lp_utils
from mava.utils.environments.flatland_utils import flatland_env_factory
from mava.utils.loggers import logger_utils
from mava.utils.offline import tfrecord_transition_dataset

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "mava_id",
    str(datetime.now()),
    "Experiment identifier that can be used to continue experiments.",
)
flags.DEFINE_string("base_dir", "logs/", "Base dir to store experiments.")


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

    # Make environment spec for dataset factory
    environment = flatland_env_factory(
        env_config=flatland_env_config, include_agent_info=False
    )
    environment_spec = MAEnvironmentSpec(environment=environment)

    # Dataset factory
    dataset_factory = functools.partial(
        tfrecord_transition_dataset,
        path="logs/datasets/tfrecords",
        environment_spec=environment_spec,
        shuffle_buffer_size=2_000_000,
    )

    # Networks factory
    network_factory = lp_utils.partial_kwargs(
        bc.make_default_networks, network_layer_sizes=(128,)
    )

    # Checkpointer appends "Checkpoints" to checkpoint_dir
    checkpoint_dir = f"{FLAGS.base_dir}/{FLAGS.mava_id}"

    # Log every [log_every] seconds.
    log_every = 10
    logger_factory = functools.partial(
        logger_utils.make_logger,
        directory=FLAGS.base_dir,
        to_terminal=True,
        to_tensorboard=True,
        time_stamp=FLAGS.mava_id,
        time_delta=log_every,
    )

    # Build distributed program
    program = bc.BC(
        dataset_factory=dataset_factory,
        environment_factory=environment_factory,
        network_factory=network_factory,
        logger_factory=logger_factory,
        optimizer=snt.optimizers.Adam(learning_rate=1e-3),
        checkpoint_subpath=checkpoint_dir,
        batch_size=32,
        executor_variable_update_period=100,
    ).build()

    # Ensure only trainer runs on gpu, while other processes run on cpu.
    local_resources = lp_utils.to_device(
        program_nodes=program.groups.keys(), nodes_on_gpu=["trainer"]
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