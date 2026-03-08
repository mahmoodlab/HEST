from __future__ import annotations

import tempfile
import warnings
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
import tifffile

from trident import load_wsi as trident_load_wsi
from trident.IO import mask_to_gdf
from trident.segmentation_models import apply_otsu_thresholding
from trident.segmentation_models.load import segmentation_model_factory
from trident.wsi_objects.WSI import WSI


class CucimWarningSingleton:
    _warned = False

    @classmethod
    def warn(cls) -> None:
        if cls._warned:
            return
        cls._warned = True
        warnings.warn(
            "CuCIM backend is not available; falling back to default WSI readers.",
            RuntimeWarning,
            stacklevel=2,
        )


def _array_to_wsi(img: np.ndarray, mpp: Optional[float] = None) -> WSI:
    arr = np.asarray(img)
    if arr.ndim != 3:
        raise ValueError("WSI ndarray input must be HxWxC")
    if arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.transpose(arr, axes=(1, 2, 0))
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if mpp is None:
        mpp = 1.0

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name
    tifffile.imwrite(tmp_path, arr)
    wsi = trident_load_wsi(tmp_path, reader_type="image", mpp=float(mpp))
    setattr(wsi, "_hest_temp_path", tmp_path)
    if hasattr(wsi, "_lazy_initialize"):
        wsi._lazy_initialize()
    return wsi


def _ensure_initialized(wsi: WSI) -> WSI:
    if hasattr(wsi, "_lazy_initialize"):
        wsi._lazy_initialize()
    return wsi


def wsi_factory(
    img: Union[str, Path, np.ndarray, WSI, Any],
    *,
    mpp: Optional[float] = None,
    reader_type: Optional[str] = None,
) -> WSI:
    if isinstance(img, WSI):
        return _ensure_initialized(img)
    if isinstance(img, np.ndarray):
        return _array_to_wsi(img, mpp=mpp)
    if isinstance(img, (str, Path)):
        try:
            loaded = trident_load_wsi(str(img), reader_type=reader_type)
        except ValueError as exc:
            if mpp is not None:
                loaded = trident_load_wsi(str(img), reader_type="image", mpp=mpp)
            else:
                raise exc
        return _ensure_initialized(loaded)
    if hasattr(img, "read_region") and hasattr(img, "dimensions"):
        width, height = img.dimensions
        arr = np.array(img.read_region((0, 0), 0, (width, height)).convert("RGB"))
        return _array_to_wsi(arr, mpp=mpp)
    raise ValueError(f"Unsupported WSI input type: {type(img)}")


def wsi_to_numpy(wsi: WSI) -> np.ndarray:
    wsi = _ensure_initialized(wsi)
    width, height = wsi.get_dimensions()
    return wsi.read_region((0, 0), 0, (width, height), read_as="numpy")


def segment_tissue_deep(
    wsi: WSI,
    pixel_size: float,
    fast_mode: bool = False,
    target_pxl_size: float = 1.0,
    patch_size_um: int = 512,
    model_name: str = "deeplabv3_seg_v4.ckpt",
    batch_size: int = 8,
    auto_download: bool = True,
    num_workers: int = 8,
    weights_dir: Optional[str] = None,
    holes_are_tissue: bool = True,
    verbose: bool=True
):
    _ = (pixel_size, patch_size_um, model_name, auto_download, weights_dir)
    effective_pxl_size = 2.0 if fast_mode and target_pxl_size == 1 else float(target_pxl_size)
    target_mag = max(1, int(round(10.0 / effective_pxl_size)))
    model = segmentation_model_factory("hest")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return wsi.segment_tissue(
        segmentation_model=model,
        target_mag=target_mag,
        job_dir=None,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        holes_are_tissue=holes_are_tissue,
        verbose=verbose,
    )
