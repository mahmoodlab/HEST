# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""Base classes for feature extraction."""

from abc import abstractmethod
from typing import Callable, Dict
from typing import Union, List, Tuple

import numpy as np
import torch
from torch import nn
from PIL.Image import Image


def prepare_module(
    module: torch.nn.Module,
    gpu: Union[None, int, List[int]] = None,
) -> Tuple[torch.nn.Module, str]:
    """Prepare ``torch.nn.Module`` by:
    - setting it to eval mode
    - disabling gradients
    - moving it to the correct device(s)

    Parameters
    ----------
    module: torch.nn.Module
        Module to parallelize data loading on.
    gpu: Union[None, int, List[int]] = None
        GPUs to use.
        If None, will use all available GPUs.
        If -1, extraction will run on CPU.

    Returns
    -------
    torch.nn.Module, str
    """
    if gpu == -1 or not torch.cuda.is_available():
        device = "cpu"
    else:
        if isinstance(gpu, int):
            device = f"cuda:{gpu}"
        else:
            # Use DataParallel to distribute the module on all GPUs
            device = "cuda:0" if gpu is None else f"cuda:{gpu[0]}"
            module = torch.nn.DataParallel(module, gpu)

    module.to(device)
    module.eval()
    module.requires_grad_(False)

    return module, device


class Extractor(nn.Module):
    """Base Extractor class."""

    @property
    def transform(self) -> Callable[[Image], torch.Tensor]:
        """
        Transform method to apply element wise.
        Default is identity.

        Returns
        -------
        transform: Callable[[Image], torch.Tensor]
        """
        return lambda x: x

    @abstractmethod
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Returns final features.

        Parameters
        ----------
        images: torch.Tensor
            (BS, C, H, W)

        Returns
        -------
        features : torch.Tensor
            (BS, D)
        """
        raise NotImplementedError

    def extract_feature_maps(
        self, images: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Returns a dictionary with intermediate features.

        Parameters
        ----------
        images: torch.Tensor
            (BS, C, H, W)

        Returns
        -------
        features : Dict[str, torch.Tensor]
            example shapes: {
                "layer4": (BS, 256, 56, 56),
                "layer5": (BS, 512, 28, 28),
                "layer6": (BS, 1024, 14, 14),
                "layer7": (BS, 2048, 7, 7),
            }
        """
        raise NotImplementedError

    def extract_features_as_numpy(self, images: torch.Tensor) -> np.ndarray:
        """Returns features as a numpy array.

        Parameters
        ----------
        images: torch.Tensor
            (BS, C, H, W)

        Returns
        -------
        features: np.ndarray
            (BS, D)
        """
        features = self.__call__(images)
        return features.cpu().detach().numpy()


class ParallelExtractor:
    """Extractor class with data parallelization.

    module: Extractor
        Extractor base class.
    gpu: Union[str, List[str], List[List[str]]]
        GPUs to use.
        If None, will use all available GPUs.
        If -1, extraction will run on CPU.
    """

    def __init__(
        self,
        module: Extractor,
        gpu: Union[str, List[str], List[List[str]]],
    ) -> None:
        self.module = module
        if "cuda:" in str(gpu):
            gpu = int(gpu.split("cuda:")[-1])
        self.feature_extractor, device = prepare_module(
            self.module.feature_extractor, gpu
        )
        device = [device]
        self.module.device = self.device = device
        self.transform = self.module.transform

    @abstractmethod
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Returns final features.

        Parameters
        ----------
        images: torch.Tensor
            (BS, C, H, W)

        Returns
        -------
        features : torch.Tensor
            (BS, D)
        """
        return self.module(images.to(self.device[0]))

    @abstractmethod
    def extract_features_as_numpy(self, images: torch.Tensor) -> np.ndarray:
        """Returns features as a numpy array.

        Parameters
        ----------
        images: torch.Tensor
            (BS, C, H, W)

        Returns
        -------
        features: np.ndarray
            (BS, D)
        """
        features = self.__call__(
            images
        )  # pylint: disable=unnecessary-dunder-call
        return features.cpu().detach().numpy()
