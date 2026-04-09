#!/usr/bin/env python3
"""
generate_stimuli.py
===================
Generate bandpass-filtered digit stimuli for the qCSF experiment,
replicating Zheng et al. (2018) "Measuring the Contrast Sensitivity
Function Using the qCSF Method With 10 Digits."

FILTER SPECIFICATION (from paper)
----------------------------------
- Type      : Raised cosine in log-frequency space, radially isotropic
- Centre    : f0 = 3 cycles per object (cpo)
- Bandwidth : 1 octave full-width at half-height (FWHH)
- Passband  : f0 / 2^(bw/2)  to  f0 * 2^(bw/2)  =  ~2.12 – 4.24 cpo
              (support extends ±bw octaves from f0, i.e. 1.5 – 6 cpo)

FREQUENCY / VISUAL ANGLE CORRESPONDENCE
-----------------------------------------
The paper states (Zheng et al. 2019 companion paper, same stimuli):
  Size (°)  :  12     6     3    1.5   0.75   0.38
  SF (cpd)  :  0.5    1     2     4     8     15.8

This gives SF_cpd = f0_eff / size_deg where f0_eff ≈ 6 cpo.
The factor-of-2 discrepancy relative to the paper's stated "3 cpo" most
likely reflects a convention in which "object" = half the image width
(the digit occupies the central 50% of the stimulus aperture, with
grey padding on each side – common practice to reduce edge effects).

In this code the filter is always defined in *cycles per image width*.
Set f0_cpo=3 if your images already contain the digit padded to 2×
its own size; set f0_cpo=6 if the digit fills the full image width.
The default is f0_cpo=3 to match the paper's stated value.

USAGE
-----
  python generate_stimuli.py <input_dir> <output_dir> [options]

  Options:
    --f0        Filter centre frequency in cpo          [default: 3.0]
    --bw        Filter bandwidth in octaves (FWHH)      [default: 1.0]
    --ppd       Display pixels per degree               [default: 60.0]
    --contrast  RMS contrast of output images (0–1)     [default: 0.20]
    --mean-lum  Background mean luminance (0–255)       [default: 128.0]
    --no-resize Skip per-condition resized outputs
    --plot      Save diagnostic plots

DEPENDENCIES
------------
  numpy, Pillow, matplotlib  (all standard; install via pip)
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Constants: SF conditions from Zheng et al. (2018)
# ─────────────────────────────────────────────────────────────────────────────

SF_CONDITIONS = [
    # (label,          size_deg,  sf_cpd)
    ("12deg_0.50cpd",  12.00,     0.50),
    ("06deg_1.00cpd",   6.00,     1.00),
    ("03deg_2.00cpd",   3.00,     2.00),
    ("1.5deg_4.00cpd",  1.50,     4.00),
    ("0.75deg_8.00cpd", 0.75,     8.00),
    ("0.38deg_15.8cpd", 0.38,    15.80),
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Filter construction
# ─────────────────────────────────────────────────────────────────────────────

def make_raised_cosine_filter(N: int,
                               f0_cpo: float = 3.0,
                               bw_octaves: float = 1.0) -> np.ndarray:
    """
    Build a 2-D radially isotropic raised cosine bandpass filter.

    Mathematical definition
    ~~~~~~~~~~~~~~~~~~~~~~~
    Let f  = radial spatial frequency in cycles per image width (cpo),
        f0 = centre frequency,
        B  = full bandwidth at half-height (octaves).

    The raised cosine in log-frequency space is:

        H(f) = ½ · (1 + cos(π · log₂(f/f0) / B))   if |log₂(f/f0)| ≤ B
             = 0                                      otherwise

    Properties
    ~~~~~~~~~~
    • H(f0)       = 1          (unit gain at centre)
    • H(f0 · 2^½B) = 0.5      (FWHH at ±B/2 octaves → full BW = B octaves) ✓
    • H(f0 · 2^B) = 0          (filter reaches zero at ±B octaves)
    • The filter is radially symmetric (depends only on |f|, not direction)
    • DC component (f=0) is always zero → output has zero mean

    Parameters
    ----------
    N         : Image side-length in pixels (square images assumed).
    f0_cpo    : Centre frequency in cycles per image width.
    bw_octaves: Full bandwidth at half-height (FWHH) in octaves.

    Returns
    -------
    H : ndarray, shape (N, N), float64
        Filter in *FFT order* (not fftshifted), ready to multiply with
        the output of np.fft.fft2().
    """
    # Frequencies in cycles per image width (cpo)
    # np.fft.fftfreq gives cycles per pixel → multiply by N → cpo
    fx = np.fft.fftfreq(N) * N   # shape (N,), FFT order
    fy = np.fft.fftfreq(N) * N
    FX, FY = np.meshgrid(fx, fy, indexing='xy')  # (N, N)
    R = np.sqrt(FX ** 2 + FY ** 2)               # radial freq, cpo

    H = np.zeros_like(R)
    nonzero = R > 0                               # exclude DC (f=0)
    log_r = np.log2(R[nonzero] / f0_cpo)

    # Raised cosine: non-zero where |log2(f/f0)| ≤ bw_octaves
    within = np.abs(log_r) <= bw_octaves
    H[nonzero] = np.where(
        within,
        0.5 * (1.0 + np.cos(np.pi * log_r / bw_octaves)),
        0.0,
    )
    return H


# ─────────────────────────────────────────────────────────────────────────────
# 2. Applying the filter
# ─────────────────────────────────────────────────────────────────────────────

def next_power_of_two(n: int) -> int:
    return int(2 ** np.ceil(np.log2(max(n, 1))))


def apply_bandpass_filter(img_gray: np.ndarray,
                           f0_cpo: float = 3.0,
                           bw_octaves: float = 1.0) -> np.ndarray:
    """
    Apply a raised cosine bandpass filter to a grayscale image.

    The image is zero-padded to the next power of two (for FFT efficiency
    and to reduce circular wrap-around artefacts), filtered in the Fourier
    domain, and then cropped back to its original size.

    Parameters
    ----------
    img_gray  : 2-D array (H × W), any numeric dtype.
                Non-square images are handled gracefully.
    f0_cpo    : Filter centre frequency in cycles per image width.
    bw_octaves: Filter FWHH bandwidth in octaves.

    Returns
    -------
    filtered : 2-D float64 array, same shape as img_gray.
               Zero-mean (DC is removed by the bandpass filter).
    """
    img = img_gray.astype(np.float64)
    H, W = img.shape
    N = next_power_of_two(max(H, W))

    # Reflect-pad to N×N to reduce edge discontinuities
    pad_h = N - H
    pad_w = N - W
    img_padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')

    # Build filter at padded size
    filt = make_raised_cosine_filter(N, f0_cpo, bw_octaves)

    # Forward FFT → multiply → inverse FFT
    F = np.fft.fft2(img_padded)
    filtered_padded = np.real(np.fft.ifft2(F * filt))

    # Crop back to original size
    return filtered_padded[:H, :W]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Normalisation
# ─────────────────────────────────────────────────────────────────────────────

def to_display_image(filtered: np.ndarray,
                     mean_lum: float = 128.0,
                     rms_contrast: float = 0.20) -> np.ndarray:
    """
    Map a zero-mean filtered image to an 8-bit display image.

    The RMS contrast is set relative to mean_lum:
        contrast = std(luminance) / mean_lum

    Output is clipped to [0, 255] and cast to uint8.

    Parameters
    ----------
    filtered     : Zero-mean float array from apply_bandpass_filter().
    mean_lum     : Mean/background luminance on the 0–255 scale.
    rms_contrast : Desired RMS contrast (0–1).
    """
    rms = filtered.std()
    if rms < 1e-12:
        return np.full(filtered.shape, mean_lum, dtype=np.uint8)
    # Scale so that std(output) = rms_contrast * mean_lum
    scaled = filtered * (rms_contrast * mean_lum / rms)
    return np.clip(mean_lum + scaled, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Resizing for each spatial-frequency condition
# ─────────────────────────────────────────────────────────────────────────────

def resize_image(img: np.ndarray, target_px: int) -> np.ndarray:
    """Resize a square uint8 image to target_px × target_px using Lanczos."""
    pil_img = Image.fromarray(img)
    resized = pil_img.resize((target_px, target_px), Image.LANCZOS)
    return np.array(resized)


def size_deg_to_pixels(size_deg: float, pixels_per_degree: float) -> int:
    """Convert visual angle to pixel count."""
    return max(4, round(size_deg * pixels_per_degree))


# ─────────────────────────────────────────────────────────────────────────────
# 5. Diagnostic plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_filter_profile(f0_cpo: float = 3.0,
                         bw_octaves: float = 1.0,
                         N: int = 512,
                         save_path: str = "filter_profile.png") -> None:
    """
    Save a two-panel figure:
      Left  – 1-D cross-section of the filter with FWHH annotation.
      Right – 2-D heat-map of the filter (fftshifted).
    """
    H = make_raised_cosine_filter(N, f0_cpo, bw_octaves)
    H_shifted = np.fft.fftshift(H)
    fx = np.fft.fftshift(np.fft.fftfreq(N) * N)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left panel: 1-D profile ──────────────────────────────────────────────
    ax = axes[0]
    profile = H_shifted[N // 2, :]          # horizontal cross-section

    ax.plot(fx, profile, 'steelblue', lw=2.5, label='H(f)')
    ax.axhline(0.5, color='grey', ls='--', lw=1.2, label='Half-height (0.5)')
    ax.axvline( f0_cpo, color='tomato', ls=':', lw=1.5, label=f'f₀ = {f0_cpo} cpo')
    ax.axvline(-f0_cpo, color='tomato', ls=':', lw=1.5)

    # Mark FWHH edges
    fwhh_lo = f0_cpo * 2 ** (-bw_octaves / 2)
    fwhh_hi = f0_cpo * 2 ** ( bw_octaves / 2)
    ax.axvline( fwhh_hi, color='seagreen', ls='-.', lw=1.2,
                label=f'FWHH edges ({fwhh_lo:.2f} / {fwhh_hi:.2f} cpo)')
    ax.axvline(-fwhh_hi, color='seagreen', ls='-.', lw=1.2)
    ax.axvline( fwhh_lo, color='seagreen', ls='-.', lw=1.2)
    ax.axvline(-fwhh_lo, color='seagreen', ls='-.', lw=1.2)

    ax.set_xlabel('Spatial frequency (cycles per object)', fontsize=12)
    ax.set_ylabel('Filter gain', fontsize=12)
    ax.set_title(
        f'Raised cosine filter\n'
        f'f₀ = {f0_cpo} cpo,  FWHH = {bw_octaves} octave',
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-0.05, 1.10)
    ax.grid(True, alpha=0.3)

    # ── Right panel: 2-D heat-map ────────────────────────────────────────────
    ax2 = axes[1]
    lim = 8
    ext = [-lim, lim, -lim, lim]
    # Crop to ±lim cpo for display
    ci = N // 2
    half = round(lim / (N / 2) * (N // 2))
    crop = H_shifted[ci - half: ci + half, ci - half: ci + half]
    im = ax2.imshow(crop, cmap='inferno', origin='lower', extent=ext,
                    vmin=0, vmax=1)
    plt.colorbar(im, ax=ax2, label='Filter gain')
    ax2.set_title('2-D filter (fftshifted, zoomed ±8 cpo)', fontsize=12)
    ax2.set_xlabel('fx (cpo)', fontsize=12)
    ax2.set_ylabel('fy (cpo)', fontsize=12)
    # Draw circles at f0 and FWHH edges
    theta = np.linspace(0, 2 * np.pi, 300)
    for r, col, ls in [(f0_cpo, 'tomato', '-'), (fwhh_lo, 'seagreen', '--'),
                        (fwhh_hi, 'seagreen', '--')]:
        ax2.plot(r * np.cos(theta), r * np.sin(theta), color=col,
                 ls=ls, lw=1.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_filtered_examples(original: np.ndarray,
                            filtered_display: np.ndarray,
                            digit_name: str,
                            save_path: str,
                            sf_conditions=None,
                            ppd: float = 60.0) -> None:
    """
    Save a figure showing: original | filtered (native) | per-SF-condition.
    """
    n_cond = len(SF_CONDITIONS)
    fig, axes = plt.subplots(1, 2 + n_cond, figsize=(3 * (2 + n_cond), 3.5))

    def show(ax, img, title):
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        ax.set_title(title, fontsize=8)
        ax.axis('off')

    show(axes[0], original, f'Original\n({original.shape[0]}px)')
    show(axes[1], filtered_display, f'Filtered\n(native res.)')

    for i, (label, size_deg, sf_cpd) in enumerate(SF_CONDITIONS):
        tpx = size_deg_to_pixels(size_deg, ppd)
        resized = resize_image(filtered_display, tpx)
        # Display at a fixed canvas size for visual comparison
        canvas = np.full((128, 128), 128, dtype=np.uint8)
        # Centre-paste
        oh = (128 - resized.shape[0]) // 2
        ow = (128 - resized.shape[1]) // 2
        oh = max(0, oh); ow = max(0, ow)
        rh = min(resized.shape[0], 128)
        rw = min(resized.shape[1], 128)
        canvas[oh:oh+rh, ow:ow+rw] = resized[:rh, :rw]
        show(axes[2 + i], canvas,
             f'{size_deg}°\n{sf_cpd} cpd\n({tpx}px)')

    fig.suptitle(f'Digit "{digit_name}" – all SF conditions', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_digits(input_dir: str,
                   output_dir: str,
                   f0_cpo: float = 3.0,
                   bw_octaves: float = 1.0,
                   pixels_per_degree: float = 60.0,
                   rms_contrast: float = 0.20,
                   mean_lum: float = 128.0,
                   save_resized: bool = True,
                   save_plots: bool = False) -> None:
    """
    Filter all digit images and save outputs.

    Steps (per digit image)
    -----------------------
    1. Load as grayscale.
    2. Apply raised cosine bandpass filter in the Fourier domain.
    3. Normalise to the specified RMS contrast relative to mean_lum.
    4. Save the native-resolution filtered image.
    5. (Optional) Resize to each SF condition and save.
    6. (Optional) Save diagnostic figure.

    Parameters
    ----------
    input_dir        : Directory containing digit PNG/TIFF/BMP images.
    output_dir       : Destination for output images.
    f0_cpo           : Filter centre frequency (cycles per image width).
    bw_octaves       : Filter FWHH bandwidth in octaves.
    pixels_per_degree: Display pixels per degree of visual angle.
    rms_contrast     : Desired RMS contrast of output images (0–1).
    mean_lum         : Mean/background luminance on the 0–255 scale.
    save_resized     : If True, save resized images for each SF condition.
    save_plots       : If True, save per-digit diagnostic figures.
    """
    input_path  = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find images
    exts = ['*.png', '*.PNG']
    image_files = []
    for ext in exts:
        image_files.extend(input_path.glob(ext))
    image_files = sorted(set(image_files))

    if not image_files:
        print(f"[ERROR] No images found in '{input_dir}'.")
        sys.exit(1)

    print(f"Found {len(image_files)} image file(s) in '{input_dir}'.")
    print(f"Filter: f0={f0_cpo} cpo,  BW={bw_octaves} oct (FWHH),  "
          f"passband: {f0_cpo * 2**(-bw_octaves/2):.2f}–"
          f"{f0_cpo * 2**(bw_octaves/2):.2f} cpo (±{bw_octaves} oct support)")
    print(f"Display: {pixels_per_degree} ppd,  mean lum={mean_lum},  "
          f"RMS contrast={rms_contrast}\n")

    for img_file in image_files:
        stem = img_file.stem
        print(f"Processing: {img_file.name}")

        # 1. Load
        img_pil  = Image.open(img_file).convert('L')
        original = np.array(img_pil, dtype=np.float64)
        print(f"  Size: {original.shape[1]} × {original.shape[0]} px")

        # 2. Filter
        filtered = apply_bandpass_filter(original, f0_cpo, bw_octaves)

        # 3. Normalise → 8-bit display image
        display  = to_display_image(filtered, mean_lum, rms_contrast)

        # 4. Save native-resolution filtered image
        out_native = output_path / f"{stem}_filtered.png"
        Image.fromarray(display).save(out_native)
        print(f"  Saved: {out_native.name}")

        # 5. Resize to each SF condition
        if save_resized:
            sf_dir = output_path / stem
            sf_dir.mkdir(exist_ok=True)
            for label, size_deg, sf_cpd in SF_CONDITIONS:
                tpx = size_deg_to_pixels(size_deg, pixels_per_degree)
                resized = resize_image(display, tpx)
                out_sf = sf_dir / f"{stem}_{label}.png"
                Image.fromarray(resized).save(out_sf)
            print(f"  Saved {len(SF_CONDITIONS)} resized images in '{sf_dir.name}/'")

        # 6. Diagnostic plot
        if save_plots:
            plot_path = str(output_path / f"{stem}_examples.png")
            plot_filtered_examples(
                original.astype(np.uint8), display, stem,
                save_path=plot_path,
                ppd=pixels_per_degree,
            )

    print(f"\nDone. All outputs in: {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Command-line interface
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            'Generate bandpass-filtered digit stimuli (Zheng et al. 2018).\n'
            'Applies a raised cosine bandpass filter and optionally resizes\n'
            'to each spatial-frequency condition.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('input_dir',
                   help='Directory containing digit images (PNG/TIFF/BMP).')
    p.add_argument('output_dir',
                   help='Directory for filtered output images.')
    p.add_argument('--f0', type=float, default=3.0, metavar='CPO',
                   help='Filter centre frequency in cycles per image width '
                        '[default: 3.0]. Use 6.0 to match the paper\'s stated '
                        'SF values (0.5 cpd at 12°, etc.) when digits fill '
                        'the full image.')
    p.add_argument('--bw', type=float, default=1.0, metavar='OCT',
                   help='Filter full bandwidth at half-height in octaves '
                        '[default: 1.0].')
    p.add_argument('--ppd', type=float, default=60.0, metavar='PPD',
                   help='Display resolution in pixels per degree of visual '
                        'angle [default: 60.0]. Affects resized output sizes.')
    p.add_argument('--contrast', type=float, default=0.20, metavar='C',
                   help='Desired RMS contrast of output images, 0–1 '
                        '[default: 0.20].')
    p.add_argument('--mean-lum', type=float, default=128.0, metavar='LUM',
                   help='Background mean luminance, 0–255 [default: 128.0].')
    p.add_argument('--no-resize', action='store_true',
                   help='Skip per-condition resized outputs.')
    p.add_argument('--plot', action='store_true',
                   help='Save diagnostic figures for each digit.')
    p.add_argument('--plot-filter', action='store_true',
                   help='Save a filter profile figure and exit.')
    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.plot_filter:
        plot_filter_profile(
            f0_cpo=args.f0,
            bw_octaves=args.bw,
            save_path=str(Path(args.output_dir) / 'filter_profile.png')
            if Path(args.output_dir).exists()
            else 'filter_profile.png',
        )
        return

    process_digits(
        input_dir         = args.input_dir,
        output_dir        = args.output_dir,
        f0_cpo            = args.f0,
        bw_octaves        = args.bw,
        pixels_per_degree = args.ppd,
        rms_contrast      = args.contrast,
        mean_lum          = args.mean_lum,
        save_resized      = not args.no_resize,
        save_plots        = args.plot,
    )


if __name__ == '__main__':
    main()
