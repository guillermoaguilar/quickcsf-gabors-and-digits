"""
Spatial frequency bandpass filtering using raised cosine filters.

Matches the filter design used in:
  Zheng et al. (2018) - "Measuring the Contrast Sensitivity Function Using
      the qCSF Method With 10 Digits"
  Zheng et al. (2019) - parallel study on Sloan letters

Filter specification
--------------------
The filter is a raised cosine (cos²) bandpass in log₂-frequency space,
applied to the radial spatial frequency magnitude in the 2-D Fourier domain.

    H(f) = cos²( π/2 · log₂(f / f_c) / bw_oct )   for |log₂(f/f_c)| ≤ bw_oct
    H(f) = 0                                         otherwise

Key values at a glance (bw_oct = 1.0):

    ┌──────────────────────────┬────────┐
    │ Frequency                │ H(f)  │
    ├──────────────────────────┼────────┤
    │ f_c / 2   (−1 oct)       │  0.0  │
    │ f_c / √2  (−0.5 oct)     │  0.5  │  ← lower half-height edge
    │ f_c       (centre)       │  1.0  │  ← peak
    │ f_c · √2  (+0.5 oct)     │  0.5  │  ← upper half-height edge
    │ f_c · 2   (+1 oct)       │  0.0  │
    └──────────────────────────┴────────┘
    Full bandwidth at half-height (FBHH) = bw_oct = 1.0 octave

Frequency units: cpo vs cpd
----------------------------
The functions support two frequency units, selected by whether ``ppd`` is
provided:

  ┌─────────────────┬───────────────────────────────────────────────────────┐
  │ ppd not given   │ center_freq is in **cycles per object** (cpo).        │
  │                 │ 1 cpo = 1 full cycle across the image.                │
  │                 │ Default: center_freq=3.0 cpo (Zheng et al.).          │
  ├─────────────────┼───────────────────────────────────────────────────────┤
  │ ppd given       │ center_freq is in **cycles per degree** (cpd).        │
  │                 │ ppd = pixels per degree of your display at your       │
  │                 │ viewing distance.                                     │
  │                 │ Conversion: f_cpp = f_cpd / ppd                       │
  └─────────────────┴───────────────────────────────────────────────────────┘

The bandwidth ``bw_oct`` is always in octaves (a frequency ratio), so it is
unit-independent — the same value applies in either mode.

Computing ppd for your setup
-----------------------------
    ppd = pixels_per_cm × viewing_distance_cm × tan(1°)
        ≈ pixels_per_cm × viewing_distance_cm × 0.017455

    Or equivalently:
        ppd = (screen_width_px / screen_width_cm) × viewing_distance_cm × 0.017455

Examples
--------
Using cycles per object (original Zheng et al. parameterisation):

    >>> filt = make_raised_cosine_filter(512, 512, center_freq=3.0)
    >>> out   = apply_raised_cosine_filter(img, center_freq=3.0)

Using cycles per degree (when you know your display's ppd):

    >>> filt = make_raised_cosine_filter(512, 512, center_freq=4.0, ppd=60.0)
    >>> out   = apply_raised_cosine_filter(img, center_freq=4.0, ppd=60.0)
    >>> bank  = apply_filter_bank(img, centers=[1.0, 2.0, 4.0, 8.0], ppd=60.0)
"""

from __future__ import annotations

import numpy as np
from numpy.fft import fft2, ifft2, fftfreq


# ---------------------------------------------------------------------------
# Unit-conversion helpers (public, for use outside this module)
# ---------------------------------------------------------------------------

def cpo_to_cpd(freq_cpo: float, image_size_px: int | float, ppd: float) -> float:
    """
    Convert a frequency from cycles per object (cpo) to cycles per degree (cpd).

    Parameters
    ----------
    freq_cpo : float
        Frequency in cycles per object (cycles across the full image).
    image_size_px : int or float
        Image size in pixels (use the geometric mean ``sqrt(H*W)`` for
        rectangular images).
    ppd : float
        Pixels per degree of visual angle.

    Returns
    -------
    float
        Frequency in cycles per degree.
    """
    size_deg = image_size_px / ppd
    return freq_cpo / size_deg


def cpd_to_cpo(freq_cpd: float, image_size_px: int | float, ppd: float) -> float:
    """
    Convert a frequency from cycles per degree (cpd) to cycles per object (cpo).

    Parameters
    ----------
    freq_cpd : float
        Frequency in cycles per degree of visual angle.
    image_size_px : int or float
        Image size in pixels (use the geometric mean ``sqrt(H*W)`` for
        rectangular images).
    ppd : float
        Pixels per degree of visual angle.

    Returns
    -------
    float
        Frequency in cycles per object.
    """
    size_deg = image_size_px / ppd
    return freq_cpd * size_deg


# ---------------------------------------------------------------------------
# Core filter construction
# ---------------------------------------------------------------------------

def make_raised_cosine_filter(
    height: int,
    width: int,
    center_freq: float = 3.0,
    bw_oct: float = 1.0,
    *,
    ppd: float | None = None,
) -> np.ndarray:
    """
    Build a 2-D raised cosine bandpass filter in the Fourier domain.

    Parameters
    ----------
    height, width : int
        Image dimensions in pixels.
    center_freq : float
        Centre frequency of the filter.  Units depend on ``ppd``:

        * ``ppd=None`` (default): **cycles per object** (cpo).
          Default value 3.0 matches Zheng et al. 2018/2019.
        * ``ppd`` given: **cycles per degree** (cpd).

    bw_oct : float
        Full bandwidth at half-height (FBHH) in octaves (default 1.0).
        The filter is zero at ±bw_oct octaves from the centre and peaks at
        1.0 at the centre.  The half-height points are at ±bw_oct/2 octaves.
        Octaves are unit-independent — the same value applies in cpo or cpd.
    ppd : float, optional
        Pixels per degree of visual angle.  When provided, ``center_freq`` is
        interpreted as cycles per degree (cpd).  Leave as None to use cpo.

    Returns
    -------
    filt : np.ndarray, shape (height, width), dtype float64
        Real-valued filter with values in [0, 1], in unshifted FFT frequency
        order (DC at [0, 0]).  Multiply directly against ``np.fft.fft2()``
        output.

    Notes
    -----
    Internal unit conversion:

        cpo mode:  f_c [cpp] = center_freq / sqrt(height × width)
        cpd mode:  f_c [cpp] = center_freq / ppd

    Both routes produce the same physical filter when the image angular size
    and the spatial frequency are consistent.
    """
    if center_freq <= 0:
        raise ValueError(f"center_freq must be positive, got {center_freq}")
    if bw_oct <= 0:
        raise ValueError(f"bw_oct must be positive, got {bw_oct}")
    if ppd is not None and ppd <= 0:
        raise ValueError(f"ppd must be positive, got {ppd}")

    # Centre frequency in cycles per pixel -----------------------------------
    if ppd is None:
        # cpo mode
        n_eff = np.sqrt(height * width)
        f_c_cpp = center_freq / n_eff
    else:
        # cpd mode: cycles/degree ÷ pixels/degree = cycles/pixel
        f_c_cpp = center_freq / ppd

    # 2-D radial frequency grid in cpp (unshifted — DC at [0,0]) ------------
    fy = fftfreq(height)
    fx = fftfreq(width)
    FX, FY = np.meshgrid(fx, fy)
    F = np.sqrt(FX**2 + FY**2)

    # log₂(f / f_c): DC (F=0) is outside every passband → set to inf --------
    with np.errstate(divide="ignore", invalid="ignore"):
        log2_ratio = np.where(F > 0, np.log2(F / f_c_cpp), np.inf)

    # Passband mask and filter -----------------------------------------------
    in_band = np.abs(log2_ratio) <= bw_oct
    log2_clamped = np.clip(log2_ratio, -bw_oct, bw_oct)
    filt = np.where(
        in_band,
        np.cos(np.pi / 2.0 * log2_clamped / bw_oct) ** 2,
        0.0,
    )
    return filt


# ---------------------------------------------------------------------------
# Convenience: apply filter to an image array
# ---------------------------------------------------------------------------

def apply_raised_cosine_filter(
    image: np.ndarray,
    center_freq: float = 3.0,
    bw_oct: float = 1.0,
    *,
    ppd: float | None = None,
    dc_offset: bool = True,
) -> np.ndarray:
    """
    Apply a raised cosine bandpass filter to a greyscale image array.

    Parameters
    ----------
    image : np.ndarray, shape (H, W)
        Greyscale image as a 2-D float array (any value range).
    center_freq : float
        Centre frequency.  See ``make_raised_cosine_filter`` for unit details.
        Default 3.0 cpo (Zheng et al.).
    bw_oct : float
        Full bandwidth at half-height in octaves (default 1.0).
    ppd : float, optional
        Pixels per degree.  When provided, ``center_freq`` is in cpd.
    dc_offset : bool
        If True (default), add 0.5 so the output sits on a mid-grey
        background (Zheng et al. convention for mean-luminance display).

    Returns
    -------
    filtered : np.ndarray, shape (H, W), dtype float64
        Filtered image.  Not clipped — clip to your display range if needed,
        e.g. ``np.clip(filtered, 0, 1)`` for [0, 1] float images.
    """
    image = np.asarray(image, dtype=float)
    h, w = image.shape
    filt = make_raised_cosine_filter(h, w, center_freq=center_freq,
                                     bw_oct=bw_oct, ppd=ppd)
    filtered = np.real(ifft2(fft2(image) * filt))
    if dc_offset:
        filtered = filtered + 0.5
    return filtered


# ---------------------------------------------------------------------------
# Batch helper: filter at multiple centre frequencies
# ---------------------------------------------------------------------------

def apply_filter_bank(
    image: np.ndarray,
    centers: list[float],
    bw_oct: float = 1.0,
    *,
    ppd: float | None = None,
    dc_offset: bool = True,
) -> dict[float, np.ndarray]:
    """
    Apply a bank of raised cosine filters at multiple centre frequencies.

    Parameters
    ----------
    image : np.ndarray, shape (H, W)
        Input greyscale image.
    centers : list of float
        Centre frequencies.  Units depend on ``ppd`` (see below).
    bw_oct : float
        Bandwidth shared by all filters in the bank (default 1.0 octave).
    ppd : float, optional
        Pixels per degree.  When provided, ``centers`` are in cpd.
    dc_offset : bool
        Passed to ``apply_raised_cosine_filter``.

    Returns
    -------
    results : dict  {center_freq: filtered_image}
        Keys match the values in ``centers``.
    """
    return {
        fc: apply_raised_cosine_filter(image, center_freq=fc, bw_oct=bw_oct,
                                       ppd=ppd, dc_offset=dc_offset)
        for fc in centers
    }
