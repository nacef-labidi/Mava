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
"""Behaviour Cloning imports.

We implement behaviour cloning like in Acme.
"""
from mava.systems.tf.offline.bc.networks import make_default_networks
from mava.systems.tf.offline.bc.system import BC
from mava.systems.tf.offline.bc.training import BCTrainer
