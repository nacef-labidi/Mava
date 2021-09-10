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

"""Utilty to load tfrecord files into a tf dataset."""
import functools
import glob
from typing import Dict

import reverb
import tensorflow as tf

from mava.specs import MAEnvironmentSpec
from mava.types import OLT, Transition


def tfrecord_transition_dataset(
    path: str,
    environment_spec: MAEnvironmentSpec,
    shuffle_buffer_size: int = 100000,
) -> tf.data.Dataset:
    """TF dataset of SARS tuples.

    This function will look for files with the file extension
    ".tfrecord" in the directory given by "path" and will return
    a tf Dataset backed by the ".tfrecord" files.

    Args:
        path: The path to the .tfrecord files. For example "~/offlinedata/mydataset"
        environment_spec: the multi-agent environment specification.
            It should be the same spec as the one used when the
            data was stored to the .tfrecord files.
        num_shards: the number of .tfrecord files.
        shuffle_buffer_size: the size of the shuffle buffer. Larger
            size results in better shuffling.
    """
    # Define function to deserialize TFExamples
    def _tf_example_to_reverb_sample(
        tf_example: tf.train.Example,
    ) -> reverb.ReplaySample:
        """Deserializes a TFExample into a reverb sample.

        Args:
            tf_example: A TFExample.

        Returns:
            reverb.ReplaySample: A reverb sample containing the
                deserialized data from the TFExample.
        """
        # Agent list and specs
        agent_list = environment_spec.get_agent_ids()
        agent_specs = environment_spec.get_agent_specs()

        # Schema
        schema = {}
        for agent in agent_list:

            # Store agent observation.
            key = "obs_" + agent
            schema[key] = tf.io.FixedLenFeature([], dtype=tf.string)
            # Store agent action.
            key = "act_" + agent
            schema[key] = tf.io.FixedLenFeature([], dtype=tf.string)
            # Store agent reward.
            key = "rew_" + agent
            schema[key] = tf.io.FixedLenFeature([], dtype=tf.string)
            # Store agent next observation.
            key = "nob_" + agent
            schema[key] = tf.io.FixedLenFeature([], dtype=tf.string)
            # Store agent discount.
            key = "dis_" + agent
            schema[key] = tf.io.FixedLenFeature([], dtype=tf.string)

        # Process example
        content = tf.io.parse_single_example(tf_example, schema)

        # Create mava style reverb sample.
        observation = {}
        action = {}
        reward = {}
        next_observation = {}
        discount = {}
        # TODO add legals and terminal to OLT.
        for agent in agent_list:

            # Store agent observation.
            key = "obs_" + agent
            obs = content[key]
            dtype = agent_specs[agent].observations.observation.dtype
            obs = tf.io.parse_tensor(obs, out_type=dtype)
            observation_olt: OLT = OLT(
                observation=obs, legal_actions=None, terminal=None
            )
            observation[agent] = observation_olt
            # Store agent action.
            key = "act_" + agent
            act = content[key]
            dtype = agent_specs[agent].actions.dtype
            action[agent] = tf.io.parse_tensor(act, out_type=dtype)
            # Store agent reward.
            key = "rew_" + agent
            rew = content[key]
            dtype = agent_specs[agent].rewards.dtype
            reward[agent] = tf.io.parse_tensor(rew, out_type=dtype)
            # Store agent next observation.
            key = "nob_" + agent
            nob = content[key]
            dtype = agent_specs[agent].observations.observation.dtype
            nob = tf.io.parse_tensor(nob, out_type=dtype)
            next_observation_olt: OLT = OLT(
                observation=nob, legal_actions=None, terminal=None
            )
            next_observation[agent] = next_observation_olt
            # Store agent discount.
            key = "dis_" + agent
            dis = content[key]
            dtype = agent_specs[agent].discounts.dtype
            discount[agent] = tf.io.parse_tensor(dis, out_type=dtype)

        # Make dummy info for reverb sample
        info = reverb.SampleInfo(
            key=tf.constant(0, tf.uint64),
            probability=tf.constant(1.0, tf.float64),
            table_size=tf.constant(0, tf.int64),
            priority=tf.constant(1.0, tf.float64),
        )

        # Empty extras.
        extras: Dict = {}
        next_extras: Dict = {}

        data: Transition = Transition(
            observation=observation,
            action=action,
            reward=reward,
            discount=discount,
            next_observation=next_observation,
            extras=extras,
            next_extras=next_extras,
        )

        return reverb.ReplaySample(info=info, data=data)

    # Process TFRecord files
    filenames = glob.glob(path + "/*.tfrecord")
    
    # num_shards = len(filenames)
    # file_ds = tf.data.Dataset.from_tensor_slices(filenames)
    # file_ds = file_ds.repeat().shuffle(num_shards)
    # example_ds = file_ds.interleave(
    #     functools.partial(tf.data.TFRecordDataset, compression_type="GZIP"),
    #     cycle_length=tf.data.experimental.AUTOTUNE,
    #     block_length=5,
    # )

    example_ds = tf.data.TFRecordDataset(filenames, compression_type="GZIP")

    example_ds = example_ds.repeat().shuffle(shuffle_buffer_size)

    # Return TFDataset backed by TFRecord files.
    return example_ds.map(
        _tf_example_to_reverb_sample, #num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
