import os
import time
from typing import Dict

import tensorflow as tf
from acme.tf import utils as tf2_utils
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import AgentRenderVariant, RenderTool

from mava import specs as mava_specs
from mava.components.tf.architectures import DecentralisedValueActor
from mava.systems.tf import madqn
from mava.systems.tf import savers as tf2_savers
from mava.utils.environments.flatland_utils import flatland_env_factory
from mava.wrappers.flatland import normalize_observation

checkpoint_subpath = "logs/2021-08-13 10:45:30.244723/"
agent_type = "train"

# flatland environment config
rail_gen_cfg: Dict = {
    "max_num_cities": 4,
    "max_rails_between_cities": 2,
    "max_rails_in_city": 3,
    "grid_mode": False,
    "seed": 0,
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

env = flatland_env_factory(env_config=flatland_env_config, include_agent_info=False)
environment_spec = mava_specs.MAEnvironmentSpec(env)

agents = environment_spec.get_agent_ids()
agent_net_keys = {agent: agent.split("_")[0] for agent in agents}

networks = madqn.make_default_networks(
    environment_spec=environment_spec,
    agent_net_keys=agent_net_keys,
    policy_networks_layer_sizes=(256, 256),
)

# architecture args
architecture_config = {
    "environment_spec": environment_spec,
    "value_networks": networks["q_networks"],
    "agent_net_keys": agent_net_keys,
}

# Create system architecture with target networks.
architecture = DecentralisedValueActor(**architecture_config)

# Create the policy_networks
networks = architecture.create_actor_variables()
before_sum = networks["values"][agent_type].variables[1].numpy().sum()

objects_to_save = {
    "q_network": networks["values"][agent_type],
}

subdir = os.path.join("trainer", agent_type)
checkpointer = tf2_savers.Checkpointer(
    time_delta_minutes=2,
    directory=checkpoint_subpath,
    objects_to_save=objects_to_save,
    subdirectory=subdir,
)

after_sum = networks["values"][agent_type].variables[1].numpy().sum()

assert before_sum != after_sum

q_network = networks["values"][agent_type]

print(q_network)

env = RailEnv(**flatland_env_config)

obs = env.reset()

# Initiate the renderer
env_renderer = RenderTool(
    env,
    agent_render_variant=AgentRenderVariant.ONE_STEP_BEHIND,
    show_debug=False,
    screen_height=600,  # Adjust these parameters to fit your resolution
    screen_width=800,
)  # Adjust these parameters to fit your resolution


class Agent:
    def __init__(self):
        pass

    def act(self, state):
        """
        :param state: input is the observation of the agent
        :return: returns an action
        """
        return tf.math.argmax(q_network(state), axis=1)


agent = Agent()

# Empty dictionary for all agent action
action_dict = dict()

print("Start episode...")
# Reset environment and get initial observations for all agents
start_reset = time.time()
obs, info = env.reset()
end_reset = time.time()
print(end_reset - start_reset)
print(
    env.get_num_agents(),
)
# Reset the rendering sytem
env_renderer.reset()

# Here you can also further enhance the provided observation by means of normalization
# See training navigation example in the baseline repository

score = 0
# Run episode
frame_step = 0
for step in range(500):
    # Chose an action for each agent in the environment
    observations = {}
    for id, observation in obs.items():
        observation = normalize_observation(observation, tree_depth=2)
        observation = tf2_utils.add_batch_dim(observation)
        action = agent.act(observation)
        action_dict.update({id: action})

    # Environment step which returns the observations for all agents, their corresponding
    # reward and whether their are done
    next_obs, all_rewards, done, _ = env.step(action_dict)
    env_renderer.render_env(show=True, show_observations=False, show_predictions=False)
    frame_step += 1

    obs = next_obs.copy()
    if done["__all__"]:
        break

print("Episode: Steps {}\t Score = {}".format(step, score))
