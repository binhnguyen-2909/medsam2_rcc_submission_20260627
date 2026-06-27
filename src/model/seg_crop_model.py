"""SegResNet crop segmenter used by the RCC box-to-mask deliverable."""

from __future__ import annotations

import functools

import cv2
import numpy as np

SIZE = 512


@functools.lru_cache(maxsize=1)
def gabor_bank() -> tuple[np.ndarray, ...]:
    kernels = []
    for theta in np.arange(0, np.pi, np.pi / 4):
        for lam in (6.0, 12.0):
            kernel = cv2.getGaborKernel((15, 15), 3.0, theta, lam, 0.5, 0, ktype=cv2.CV_32F)
            kernel -= kernel.mean()
            kernels.append(kernel)
    return tuple(kernels)


def texture_chan(gray: np.ndarray) -> np.ndarray:
    gray_f = gray.astype(np.float32) / 255.0
    acc = None
    for kernel in gabor_bank():
        response = np.abs(cv2.filter2D(gray_f, cv2.CV_32F, kernel))
        acc = response if acc is None else np.maximum(acc, response)
    max_v = float(acc.max())
    return acc / max_v if max_v > 1e-6 else acc


def make_channels(bgr: np.ndarray, mode: str) -> np.ndarray:
    """Return normalized CHW float32 tensor data from a BGR crop."""
    chans = []
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chans.append((rgb - 0.5) / 0.5)
    if "lab" in mode:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
        chans.append((lab - 0.5) / 0.5)
    if "hsv" in mode:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] /= 179.0
        hsv[..., 1] /= 255.0
        hsv[..., 2] /= 255.0
        chans.append((hsv - 0.5) / 0.5)
    arr = np.concatenate(chans, axis=2)
    if "tex" in mode:
        tex = texture_chan(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))[..., None]
        arr = np.concatenate([arr, tex], axis=2)
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


def n_channels(mode: str) -> int:
    channels = 3
    if "lab" in mode:
        channels += 3
    if "hsv" in mode:
        channels += 3
    if "tex" in mode:
        channels += 1
    return channels


def pad_box(box: list[float], pad: float, width: int, height: int) -> list[int]:
    x0, y0, x1, y1 = box
    box_w = x1 - x0
    box_h = y1 - y0
    x0 -= box_w * pad
    x1 += box_w * pad
    y0 -= box_h * pad
    y1 += box_h * pad
    x0 = max(0, int(round(x0)))
    y0 = max(0, int(round(y0)))
    x1 = min(width, int(round(x1)))
    y1 = min(height, int(round(y1)))
    if x1 <= x0:
        x1 = min(width, x0 + 1)
    if y1 <= y0:
        y1 = min(height, y0 + 1)
    return [x0, y0, x1, y1]


def build_model(arch: str, in_ch: int):
    from monai.networks.nets import SegResNet, SwinUNETR

    if arch == "segresnet":
        return SegResNet(
            spatial_dims=2,
            in_channels=in_ch,
            out_channels=1,
            init_filters=32,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
        )
    if arch == "swinunetr":
        return SwinUNETR(in_channels=in_ch, out_channels=1, spatial_dims=2, feature_size=24)
    raise ValueError(f"Unknown architecture: {arch}")
