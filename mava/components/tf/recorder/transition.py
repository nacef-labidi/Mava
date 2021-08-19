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

# # Adapted from
# https://github.com/deepmind/acme/blob/master/acme/adders/reverb/transition.py
from typing import List
import datetime
import dm_env
import glob

import tensorflow as tf
import reverb

"""Transition adder for TFRecord."""


class ParallelTransitionRecordWriter:
    def __init__(
        self,
        agent_list,
        transitions_per_file: int = 100_000,
        record_dir: str = "tfrecords",
    ):
        self._agent_list = agent_list
        self._record_dir = record_dir
        self._transitions_per_file = transitions_per_file
        self._observation = None
        self._buffer: List[tf.train.Example] = []
        self._ctr = 0
        self._num_writes = 0

    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        value = tf.io.serialize_tensor(value).numpy()
        bytes_list = tf.train.BytesList(value=[value])
        return tf.train.Feature(bytes_list=bytes_list)

    def add_first(self, timestep: dm_env.TimeStep):
        self._observation = timestep.observation

    def add(self, actions, next_timestep: dm_env.TimeStep):
        transition = {}
        for agent in self._agent_list:
            # NOTE we make the end code exactly 4 characters
            # including the underscore e.g. "_obs".

            # Store agent observation.
            key = agent + "_obs"
            obs = tf.convert_to_tensor(
                self._observation[agent].observation, dtype=tf.float32
            )
            transition[key] = self._bytes_feature(obs)
            # Store agent action.
            key = agent + "_act"
            act = tf.convert_to_tensor(actions[agent], dtype=tf.int32)
            transition[key] = self._bytes_feature(act)
            # Store agent reward.
            key = agent + "_rew"
            rew = tf.convert_to_tensor(next_timestep.reward[agent], dtype=tf.float32)
            transition[key] = self._bytes_feature(rew)
            # Store agent next observation.
            key = agent + "_nob"
            nob = tf.convert_to_tensor(
                next_timestep.observation[agent].observation, dtype=tf.float32
            )
            transition[key] = self._bytes_feature(nob)
            # Store agent discount.
            key = agent + "_dis"
            dis = tf.convert_to_tensor(next_timestep.discount[agent], dtype=tf.float32)
            transition[key] = self._bytes_feature(dis)

        transition = tf.train.Example(features=tf.train.Features(feature=transition))

        self._buffer.append(transition)
        self._ctr += 1

        self._observation = next_timestep.observation

        self._write()

    def _write(self):
        if self._ctr > 0 and (self._ctr % self._transitions_per_file == 0):
            filename = self._record_dir + "/" + str(self._num_writes) + ".tfrecord"
            writer = tf.io.TFRecordWriter(filename)
            for transition in self._buffer:
                writer.write(transition.SerializeToString())
            writer.close()
            self._num_writes += 1

            # clear buffer
            self._buffer = []


class TFRecordLoader:
    def __init__(self, agent_list, recorder_dir: str = "tfrecords"):
        self._agent_list = agent_list
        self._recorder_dir = recorder_dir

    def _decode_fn(self, record_bytes):
        # Schema
        schema = {}
        for agent in self._agent_list:

            # Store agent observation.
            key = agent + "_obs"
            schema[key] = tf.io.FixedLenFeature([], dtype=tf.string)
            # Store agent action.
            key = agent + "_act"
            schema[key] = tf.io.FixedLenFeature([], dtype=tf.string)
            # Store agent reward.
            key = agent + "_rew"
            schema[key] = tf.io.FixedLenFeature([], dtype=tf.string)
            # Store agent next observation.
            key = agent + "_nob"
            schema[key] = tf.io.FixedLenFeature([], dtype=tf.string)
            # Store agent discount.
            key = agent + "_dis"
            schema[key] = tf.io.FixedLenFeature([], dtype=tf.string)

        content = tf.io.parse_single_example(record_bytes, schema)

        observation = {}
        action = {}
        reward = {}
        next_observation = {}
        discount = {}
        for agent in self._agent_list:

            # Store agent observation.
            key = agent + "_obs"
            obs = content[key]
            observation[agent] = tf.io.parse_tensor(obs, out_type=tf.float32)
            # Store agent action.
            key = agent + "_act"
            act = content[key]
            action[agent] = tf.io.parse_tensor(act, out_type=tf.int32)
            # Store agent reward.
            key = agent + "_rew"
            rew = content[key]
            reward[agent] = tf.io.parse_tensor(rew, out_type=tf.float32)
            # Store agent next observation.
            key = agent + "_nob"
            nob = content[key]
            next_observation[agent] = tf.io.parse_tensor(nob, out_type=tf.float32)
            # Store agent discount.
            key = agent + "_dis"
            dis = content[key]
            discount[agent] = tf.io.parse_tensor(dis, out_type=tf.float32)

        info = reverb.SampleInfo(
            key=tf.constant(0, tf.uint64),
            probability=tf.constant(1.0, tf.float64),
            table_size=tf.constant(0, tf.int64),
            priority=tf.constant(1.0, tf.float64),
        )

        data = (observation, action, reward, discount, next_observation, {})

        return reverb.ReplaySample(info=info, data=data)

    def as_tf_dataset(self):
        files = glob.glob(self._recorder_dir + "/*.tfrecord")
        return tf.data.TFRecordDataset(files).map(self._decode_fn)


##### TESTS #####
# import numpy as np

# writer = ParallelTransitionRecordWriter(["agent"], transitions_per_file=2)

# # FIRST TIMESTEP
# obs = {"agent": np.array([1,1,1])}
# rew = {"agent": 100.}
# dis = {"agent": 0.99}

# timestep = dm_env.TimeStep(
#     observation=obs,
#     reward=rew,
#     discount=dis,
#     step_type=dm_env.StepType.FIRST,
# )

# writer.add_first(timestep)

# # SECOND TIMESTEP
# obs = {"agent": np.array([2,2,2])}
# act = {"agent": 2}
# rew = {"agent": 200.}
# dis = {"agent": 0.98}

# timestep = dm_env.TimeStep(
#     observation=obs,
#     reward=rew,
#     discount=dis,
#     step_type=dm_env.StepType.MID,
# )

# writer.add(act, timestep)

# # THIRD TIMESTEP
# obs = {"agent": np.array([3,3,3])}
# act = {"agent": 3}
# rew = {"agent": 300.}
# dis = {"agent": 0.97}

# timestep = dm_env.TimeStep(
#     observation=obs,
#     reward=rew,
#     discount=dis,
#     step_type=dm_env.StepType.MID,
# )

# writer.add(act, timestep)

# # The writer should have written to a file now.

# loader = TFRecordLoader(["agent"])

# dataset = loader.as_tf_dataset()

# sample = next(iter(dataset))

# data = sample.data

# observation, action, reward, discount, next_observation, extras = data

# print(observation["agent"])
# print(next_observation["agent"])
# print(action["agent"])
