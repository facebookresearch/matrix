# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from matrix.cluster.ray_cluster import _normalize_slurm_keys


def test_normalize_slurm_keys_aliases():
    cfg = {"slurm_account": "acc", "slurm_qos": "q"}
    assert _normalize_slurm_keys(cfg) == {"account": "acc", "qos": "q"}


def test_normalize_slurm_keys_passthrough():
    cfg = {"gpus_per_node": 8}
    assert _normalize_slurm_keys(cfg) == cfg
