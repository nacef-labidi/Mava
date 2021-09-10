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
from mava.systems.tf.offline import madqn
from mava.utils import lp_utils
from mava.utils.environments.flatland_utils import flatland_env_factory
from mava.utils.loggers import logger_utils
from mava.utils.offline import tfrecord_transition_dataset


# Flatland environment config
rail_gen_cfg: Dict = {
    "max_num_cities": 2,
    "max_rails_between_cities": 2,
    "max_rails_in_city": 3,
    "grid_mode": False,
    "seed": 42,
}

flatland_env_config: Dict = {
    "number_of_agents": 2,
    "width": 25,
    "height": 25,
    "rail_generator": sparse_rail_generator(**rail_gen_cfg),
    "schedule_generator": sparse_schedule_generator(),
    "obs_builder_object": TreeObsForRailEnv(
        max_depth=2, predictor=ShortestPathPredictorForRailEnv(max_depth=30)
    ),
}

# Make environment spec for dataset factory
environment = flatland_env_factory(
    env_config=flatland_env_config, include_agent_info=False
)
environment_spec = MAEnvironmentSpec(environment=environment)

# Dataset factory
dataset = tfrecord_transition_dataset(
path="tfrecords/2021-09-07 09:07:58.744968",
environment_spec=environment_spec,
shuffle_buffer_size=100_000,
)

# dataset = dataset.batch(200)

# print(len(dataset))

# print(next(dataset).data.observation["train_0"].observation[0])

# print(environment.reset())

i = 0
for e in dataset:
    i += 1
    if i % 10_000 == 0:
        print(i)

# print(i)