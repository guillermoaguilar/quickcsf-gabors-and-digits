#!/usr/bin/env python3
"""
generate_stimuli.py
===================
Generate bandpass-filtered digit stimuli for the qCSF experiment,
replicating Zheng et al. (2018) "Measuring the Contrast Sensitivity
Function Using the qCSF Method With 10 Digits."

APPROACH
--------
All output images have the same pixel dimensions as the input.  The six
spatial-frequency conditions are produced by varying the centre frequency
of the bandpass filter, not by resizing.  The mapping is:

    f0_cpo = sf_cpd × display_size_deg

where display_size_deg is the visual angle subtended by the image on your
monitor (a fixed property of your setup, set with --display-size).

Example: display_size=3°, sf_cpd=2 → f0_cpo = 6 cpo.

FILTER SPECIFICATION
--------------------
- Type      : Raised cosine in log-frequency space, radially isotropic
- Centre    : f0_cpo = sf_cpd × display_size_deg  (per SF condition)
- Bandwidth : 1 octave full-width at half-height (FWHH)

    H(f) = ½ · (1 + cos(π · log₂(f/f0) / bw))   if |log₂(f/f0)| ≤ bw
         = 0                                        otherwise

SF CONDITIONS (Zheng et al. 2018)
----------------------------------
  Target SF (cpd)  :  0.5    1     2     4     8    15.8

  The required f0_cpo depends on your display size.  For a 3° image:
    f0_cpo           :  1.5    3     6    12    24    47.4

  Important: the filter is only meaningful if f0_cpo < N/2 (Nyquist).
  A warning is printed when f0_cpo approaches or exceeds this limit.

RESIZING
--------
If your source images are larger than what you will display (e.g. 512 px
source displayed at a size that corresponds to 256 px on screen), always
filter FIRST at the full source resolution, then resize.  Never resize
before filtering -- that lowers Nyquist and may truncate high-SF filters.

Use --output-px N to Lanczos-resize every filtered image to N x N pixels
after filtering.  The filter is always computed for the source image at
--display-size degrees, so the frequency content is correct regardless of
whether you resize afterward.

Example (512 px source = 8 deg, displayed at 4 deg -> 256 px on screen):
  python generate_stimuli.py digits/ stimuli/ --display-size 4 --output-px 256

USAGE
-----
  python generate_stimuli.py <input_dir> <output_dir> [options]

  Options:
    --display-size  Visual angle of the image on screen (degrees) [default: 3.0]
    --output-px     Resize filtered images to this size in pixels  [optional]
    --bw            Filter bandwidth in octaves (FWHH)             [default: 1.0]
    --contrast      RMS contrast of output images (0-1)            [default: 0.20]
    --mean-lum      Background mean luminance (0-255)              [default: 128.0]
    --plot          Save a strip figure for each digit
    --plot-filters  Save an overlay of all six filter profiles and exit

DEPENDENCIES
------------
  numpy, Pillow, matplotlib
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# SF conditions from Zheng et al. (2018) — target cpd values are fixed;
# f0_cpo is computed at runtime from display_size_deg.
# ---------------------------------------------------------------------------

SF_CONDITIONS = [
    # (label,       sf_cpd)
    ("0.50cpd",      0.50),
    ("1.00cpd",      1.00),
    ("2.00cpd",      2.00),
    ("4.00cpd",      4.00),
    ("8.00cpd",      8.00),
    ("15.80cpd",    15.80),
]


def f0_for_condition(sf_cpd: float, display_size_deg: float) -> float:
    """Return the filter centre frequency (cpo) for a target sf_cpd.

    f0_cpo = sf_cpd × display_size_deg

    Rationale: if the image subtends display_size_deg degrees and the filter
    peak should fall at sf_cpd cycles per degree, then the peak must be at
    sf_cpd × display_size_deg cycles across the full image width.
    """
    return sf_cpd * display_size_deg


# ---------------------------------------------------------------------------
# 1. Filter construction
# ---------------------------------------------------------------------------

def make_raised_cosine_filter(N: int,
                               f0_cpo: float,
                               bw_octaves: float = 1.0) -> np.ndarray:
    """Build a 2-D radially isotropic raised cosine bandpass filter.

    Parameters
    ----------
    N         : Side-length of the (square) padded image in pixels.
    f0_cpo    : Centre frequency in cycles per image width.
    bw_octaves: Full bandwidth at half-height (FWHH) in octaves.

    Returns
    -------
    H : ndarray shape (N, N), float64, in FFT order (not fftshifted).
    """
    fx = np.fft.fftfreq(N) * N          # cycles per image width, FFT order
    fy = np.fft.fftfreq(N) * N
    FX, FY = np.meshgrid(fx, fy, indexing='xy')
    R = np.sqrt(FX ** 2 + FY ** 2)      # radial frequency (cpo)

    H = np.zeros_like(R)
    nonzero = R > 0
    log_r = np.log2(R[nonzero] / f0_cpo)
    within = np.abs(log_r) <= bw_octaves
    H[nonzero] = np.where(
        within,
        0.5 * (1.0 + np.cos(np.pi * log_r / bw_octaves)),
        0.0,
    )
    return H


# ---------------------------------------------------------------------------
# 2. Applying the filter
# ---------------------------------------------------------------------------

def next_power_of_two(n: int) -> int:
    return int(2 ** np.ceil(np.log2(max(n, 1))))


def apply_bandpass_filter(img_gray: np.ndarray,
                           f0_cpo: float,
                           bw_octaves: float = 1.0) -> np.ndarray:
    """Apply a raised cosine bandpass filter to a grayscale image.

    The image is reflect-padded to the next power of two before FFT to
    reduce circular wrap-around artefacts, then cropped back.

    Parameters
    ----------
    img_gray  : 2-D array (H x W), any numeric dtype.
    f0_cpo    : Filter centre frequency in cycles per image width.
    bw_octaves: Filter FWHH bandwidth in octaves.

    Returns
    -------
    filtered : 2-D float64 array, same shape as img_gray, zero mean.
    """
    img = img_gray.astype(np.float64)
    H, W = img.shape
    N = next_power_of_two(max(H, W))

    img_padded = np.pad(img, ((0, N - H), (0, N - W)), mode='reflect')
    filt = make_raised_cosine_filter(N, f0_cpo, bw_octaves)
    filtered_padded = np.real(np.fft.ifft2(np.fft.fft2(img_padded) * filt))

    return filtered_padded[:H, :W]


# ---------------------------------------------------------------------------
# 3. Normalisation
# ---------------------------------------------------------------------------

def to_display_image(filtered: np.ndarray,
                     mean_lum: float = 128.0,
                     rms_contrast: float = 0.20) -> np.ndarray:
    """Map a zero-mean filtered image to an 8-bit display image.

        contrast = std(luminance) / mean_lum

    Output is clipped to [0, 255] and cast to uint8.
    """
    rms = filtered.std()
    if rms < 1e-12:
        return np.full(filtered.shape, mean_lum, dtype=np.uint8)
    scaled = filtered * (rms_contrast * mean_lum / rms)
    return np.clip(mean_lum + scaled, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 4. Diagnostic plots
# ---------------------------------------------------------------------------

def plot_all_filters(display_size_deg: float,
                     bw_octaves: float = 1.0,
                     N: int = 512,
                     save_path: str = "filter_profiles.png") -> None:
    """Save a two-panel figure showing all six filter profiles.

    Left  : 1-D cross-sections of all filters overlaid.
    Right : 2-D heat-map of the highest-frequency filter (hardest case).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colours = plt.cm.viridis(np.linspace(0.1, 0.9, len(SF_CONDITIONS)))

    ax = axes[0]
    fx_ax = np.fft.fftshift(np.fft.fftfreq(N) * N)
    cx = N // 2

    for (label, sf_cpd), col in zip(SF_CONDITIONS, colours):
        f0 = f0_for_condition(sf_cpd, display_size_deg)
        H = make_raised_cosine_filter(N, f0, bw_octaves)
        profile = np.fft.fftshift(H)[cx, :]
        ax.plot(fx_ax, profile, color=col, lw=2,
                label=f'{sf_cpd} cpd  (f0={f0:.1f} cpo)')

    ax.axhline(0.5, color='grey', ls='--', lw=1, alpha=0.6, label='Half-height')
    ax.set_xlabel('Spatial frequency (cycles per image width)', fontsize=11)
    ax.set_ylabel('Filter gain', fontsize=11)
    ax.set_title(
        f'All SF conditions — display size = {display_size_deg} deg, '
        f'BW = {bw_octaves} oct',
        fontsize=11,
    )
    ax.legend(fontsize=8, loc='upper right')
    f0_max = f0_for_condition(SF_CONDITIONS[-1][1], display_size_deg)
    x_lim  = min(f0_max * 2.5, N / 2)
    ax.set_xlim(-x_lim, x_lim)
    ax.set_ylim(-0.05, 1.15)
    ax.grid(True, alpha=0.3)

    # 2-D map of the highest-frequency filter
    ax2 = axes[1]
    H_hi  = np.fft.fftshift(make_raised_cosine_filter(N, f0_max, bw_octaves))
    lim   = min(f0_max * 2.5, N / 2)
    half  = round(lim / (N / 2) * cx)
    crop  = H_hi[cx - half: cx + half, cx - half: cx + half]
    im    = ax2.imshow(crop, cmap='inferno', origin='lower',
                       extent=[-lim, lim, -lim, lim], vmin=0, vmax=1)
    plt.colorbar(im, ax=ax2, label='Filter gain')
    ax2.set_title(
        f'2-D filter for {SF_CONDITIONS[-1][1]} cpd (f0={f0_max:.1f} cpo)',
        fontsize=11,
    )
    ax2.set_xlabel('fx (cpo)'); ax2.set_ylabel('fy (cpo)')
    theta = np.linspace(0, 2 * np.pi, 300)
    for r, col, ls in [
        (f0_max,                         'white', '-'),
        (f0_max * 2**(-bw_octaves / 2),  'lime',  '--'),
        (f0_max * 2**( bw_octaves / 2),  'lime',  '--'),
    ]:
        if r <= lim:
            ax2.plot(r * np.cos(theta), r * np.sin(theta),
                     color=col, ls=ls, lw=1.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def _save_strip(original: np.ndarray,
                displays: list,
                digit_name: str,
                save_path: str) -> None:
    """Save a horizontal strip: original + one panel per SF condition."""
    n = 1 + len(displays)
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 3))

    def show(ax, img, title):
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        ax.set_title(title, fontsize=8)
        ax.axis('off')

    show(axes[0], original.astype(np.uint8),
         f'Original\n({original.shape[1]}x{original.shape[0]}px)')
    for i, ((label, sf_cpd), disp) in enumerate(zip(SF_CONDITIONS, displays)):
        show(axes[i + 1], disp, f'{sf_cpd} cpd')

    fig.suptitle(f'Digit "{digit_name}"', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# 5. Main pipeline
# ---------------------------------------------------------------------------

def process_digits(input_dir: str,
                   output_dir: str,
                   display_size_deg: float = 3.0,
                   bw_octaves: float = 1.0,
                   rms_contrast: float = 0.20,
                   mean_lum: float = 128.0,
                   output_px: int = None,
                   save_plots: bool = False) -> None:
    """Filter all digit images across all six SF conditions and save outputs.

    For each digit image and each SF condition:
      1. Compute f0_cpo = sf_cpd x display_size_deg.
      2. Warn if f0_cpo is close to the Nyquist limit of the image.
      3. Apply the raised cosine bandpass filter.
      4. Normalise to the target RMS contrast.
      5. Optionally Lanczos-resize to output_px x output_px.
      6. Save the result.

    The resize (step 5) always happens AFTER filtering so that the filter
    operates at the full source resolution, avoiding Nyquist problems.

    Output structure
    ----------------
      output_dir/
        digit_0/
          digit_0_0.50cpd.png
          digit_0_1.00cpd.png
          ...
        digit_1/
          ...

    Parameters
    ----------
    input_dir       : Directory containing digit images (PNG/TIFF/BMP/JPG).
    output_dir      : Root destination for filtered outputs.
    display_size_deg: Visual angle subtended by the image on screen (degrees).
                      Used to compute f0_cpo = sf_cpd x display_size_deg.
                      This should reflect the ACTUAL displayed visual angle,
                      not the source image's angular size if they differ.
    bw_octaves      : Filter FWHH bandwidth in octaves.
    rms_contrast    : Desired RMS contrast of output images (0-1).
    mean_lum        : Background mean luminance on the 0-255 scale.
    output_px       : If given, Lanczos-resize each filtered image to this
                      many pixels (square) AFTER filtering. Use this when
                      your source images are larger than what you will display
                      (e.g. 512 px source displayed at 256 px on screen).
                      The filter is still computed for display_size_deg.
    save_plots      : If True, save a per-digit strip figure.
    """
    input_path  = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    exts = ['*.png', '*.PNG', '*.tif', '*.tiff', '*.TIF', '*.TIFF',
            '*.bmp', '*.BMP', '*.jpg', '*.jpeg', '*.JPG', '*.JPEG']
    image_files = sorted(set(f for ext in exts for f in input_path.glob(ext)))

    if not image_files:
        print(f"[ERROR] No images found in '{input_dir}'.")
        sys.exit(1)

    print(f"Found {len(image_files)} image file(s) in '{input_dir}'.")
    resize_note = f"   output: {output_px}x{output_px} px (filter-first)" if output_px else ""
    print(f"Display size: {display_size_deg} deg   BW: {bw_octaves} oct   "
          f"RMS contrast: {rms_contrast}   mean lum: {mean_lum}{resize_note}\n")
    print(f"{'SF (cpd)':<12} {'f0 (cpo)':<12} {'Support (cpo)'}")
    print(f"{'--------':<12} {'--------':<12} {'-------------'}")
    for label, sf_cpd in SF_CONDITIONS:
        f0 = f0_for_condition(sf_cpd, display_size_deg)
        lo = f0 * 2 ** (-bw_octaves)
        hi = f0 * 2 ** ( bw_octaves)
        print(f"{sf_cpd:<12.2f} {f0:<12.3f} {lo:.3f} - {hi:.3f}")
    print()

    for img_file in image_files:
        stem = img_file.stem
        print(f"Processing: {img_file.name}")

        img_pil  = Image.open(img_file).convert('L')
        original = np.array(img_pil, dtype=np.float64)
        H_px, W_px = original.shape
        N_pad   = next_power_of_two(max(H_px, W_px))
        nyquist = N_pad / 2
        print(f"  Size: {W_px} x {H_px} px   "
              f"(FFT pad: {N_pad}, Nyquist: {nyquist:.0f} cpo)")

        sf_dir = output_path / stem
        sf_dir.mkdir(exist_ok=True)

        displays = []

        for label, sf_cpd in SF_CONDITIONS:
            f0 = f0_for_condition(sf_cpd, display_size_deg)
            support_hi = f0 * 2 ** bw_octaves

            if support_hi > nyquist:
                print(f"  [WARN] {sf_cpd} cpd: filter support upper edge "
                      f"({support_hi:.1f} cpo) exceeds Nyquist "
                      f"({nyquist:.0f} cpo). Use a higher-resolution input.")
            elif f0 > nyquist * 0.5:
                print(f"  [NOTE] {sf_cpd} cpd: f0={f0:.1f} cpo is above "
                      f"half-Nyquist; filter may be partially truncated.")

            filtered = apply_bandpass_filter(original, f0, bw_octaves)
            display  = to_display_image(filtered, mean_lum, rms_contrast)

            # Resize AFTER filtering to avoid Nyquist truncation
            if output_px is not None:
                pil_out = Image.fromarray(display).resize(
                    (output_px, output_px), Image.LANCZOS)
                display_out = np.array(pil_out)
            else:
                display_out = display

            displays.append(display_out)

            out_path = sf_dir / f"{stem}_{label}.png"
            Image.fromarray(display_out).save(out_path)

        print(f"  Saved {len(SF_CONDITIONS)} filtered images -> '{sf_dir.name}/'")

        if save_plots:
            _save_strip(original, displays, stem,
                        str(output_path / f"{stem}_strip.png"))

    print(f"\nDone. All outputs in: {output_dir}")


# ---------------------------------------------------------------------------
# 6. Command-line interface
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            'Generate bandpass-filtered digit stimuli (Zheng et al. 2018).\n'
            'Produces one image per digit per SF condition, all at the\n'
            'original pixel resolution, by varying the filter centre\n'
            'frequency rather than resizing.\n\n'
            '  f0_cpo = sf_cpd x display_size_deg'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('input_dir',
                   help='Directory containing digit images (PNG/TIFF/BMP/JPG).')
    p.add_argument('output_dir',
                   help='Root directory for filtered output images.')
    p.add_argument('--display-size', type=float, default=3.0, metavar='DEG',
                   help='Visual angle (degrees) subtended by the image on '
                        'your monitor. Determines f0_cpo = sf_cpd x DEG '
                        'for each SF condition. [default: 3.0]')
    p.add_argument('--output-px', type=int, default=None, metavar='N',
                   help='If given, Lanczos-resize each filtered image to '
                        'N x N pixels AFTER filtering. Use this when your '
                        'source images are larger than what you display '
                        '(e.g. --output-px 256 for a 512 px source displayed '
                        'at half size). The filter is always applied first '
                        'at the full source resolution.')
    p.add_argument('--bw', type=float, default=1.0, metavar='OCT',
                   help='Filter full bandwidth at half-height in octaves '
                        '[default: 1.0].')
    p.add_argument('--contrast', type=float, default=0.20, metavar='C',
                   help='Desired RMS contrast of output images, 0-1 '
                        '[default: 0.20].')
    p.add_argument('--mean-lum', type=float, default=128.0, metavar='LUM',
                   help='Background mean luminance, 0-255 [default: 128.0].')
    p.add_argument('--plot', action='store_true',
                   help='Save a strip figure for each digit.')
    p.add_argument('--plot-filters', action='store_true',
                   help='Save an overlay of all six filter profiles and exit.')
    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.plot_filters:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        plot_all_filters(
            display_size_deg = args.display_size,
            bw_octaves       = args.bw,
            save_path        = str(out / 'filter_profiles.png'),
        )
        return

    process_digits(
        input_dir        = args.input_dir,
        output_dir       = args.output_dir,
        display_size_deg = args.display_size,
        bw_octaves       = args.bw,
        rms_contrast     = args.contrast,
        mean_lum         = args.mean_lum,
        output_px        = args.output_px,
        save_plots       = args.plot,
    )


if __name__ == '__main__':
    main()
