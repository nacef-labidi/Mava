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
import os

import reverb
import tensorflow as tf

from mava.specs import MAEnvironmentSpec


def _tf_example_to_reverb_sample(tf_example: tf.train.Example) -> reverb.ReplaySample:
    """Deserializes a TFExample into a reverb sample.

    Args:
        tf_example: A TFExample.

    Returns:
        reverb.ReplaySample: A reverb sample containing the
            deserialized data from the TFExample.
    """
    pass


def tfrecord_transition_dataset(
    path: str,
    environment_spec: MAEnvironmentSpec,
    num_shards: int,
    shuffle_buffer_size: int = 100000,
) -> tf.data.Dataset:
    """TF dataset of SARS tuples.

    Args:
        path: The path to the .tfrecord files.
        environment_spec: the multi-agent environment specification.
            It should be the same spec as the one used when the
            data was stored to the .tfrecord files.
        num_shards: the number of .tfrecord files.
        shuffle_buffer_size: the size of the shuffle buffer. Larger
            results in better shuffling.
    """
    filenames = [os.path.join(path, str(i) + ".tfrecord") for i in range(num_shards)]
    file_ds = tf.data.Dataset.from_tensor_slices(filenames)
    file_ds = file_ds.repeat().shuffle(num_shards)
    example_ds = file_ds.interleave(
        functools.partial(tf.data.TFRecordDataset, compression_type="GZIP"),
        cycle_length=tf.data.experimental.AUTOTUNE,
        block_length=5,
    )
    example_ds = example_ds.shuffle(shuffle_buffer_size)
    return example_ds.map(
        _tf_example_to_reverb_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
