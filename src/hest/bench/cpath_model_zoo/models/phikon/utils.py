# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""Utility functions for feature extractors."""

from typing import List

from torch import nn
from torchvision.models.feature_extraction import get_graph_node_names


def validate_node_names(module: nn.Module, node_names: List[str]) -> None:
    """Validates network/graph node names.

    Parameters
    ----------
    module : nn.Module
        Module for which the node names will be retrieved.
    node_names : List[str]
        List of expected node names.

    Raises
    ------
    ValueError
        If one of the ``node_name`` from the module is not expected.
    """
    valid_node_names, _ = get_graph_node_names(model=module)
    for node_name in node_names:
        if node_name not in valid_node_names:
            raise ValueError(
                f"Got an invalid graph node name ({node_name}) "
                f"for the given network."
            )
