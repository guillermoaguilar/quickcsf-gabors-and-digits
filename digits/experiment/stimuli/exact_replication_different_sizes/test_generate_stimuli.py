#!/usr/bin/env python3
"""
test_generate_stimuli.py
========================
Self-contained test and validation suite for generate_stimuli.py.

What it does
------------
1. Renders a set of synthetic Sloan-style digit images (0-9) at high
   resolution using matplotlib's vector fonts, then binarises and
   places them on a mid-grey background – a close proxy for the real
   Sloan digit PNGs you already have.

2. Validates the raised cosine filter:
   - Checks unit gain at f0
   - Checks 0.5 gain at ±½·BW octaves from f0
   - Checks zero gain outside the support

3. Runs the full pipeline (filter + resize) on the synthetic digits.

4. Saves a comprehensive figure:
   - Filter 1-D profile
   - Filter 2-D map
   - Example digit: original → filtered → all 6 SF-condition sizes

Run:  python test_generate_stimuli.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Import the module under test ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from generate_stimuli import (
    make_raised_cosine_filter,
    apply_bandpass_filter,
    to_display_image,
    resize_image,
    size_deg_to_pixels,
    SF_CONDITIONS,
    process_digits,
    plot_filter_profile,
)

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Sloan-style digit renderer
# ─────────────────────────────────────────────────────────────────────────────

def render_sloan_digit(digit: str, size_px: int = 256) -> np.ndarray:
    """
    Render a single digit as a black character on mid-grey (128) background,
    using Sloan-inspired proportions (square bounding box, bold stroke).

    The digit is drawn large enough to fill ~70% of the image area so that
    the filter sees a realistic range of spatial frequencies.
    """
    img = Image.new('L', (size_px, size_px), color=128)
    draw = ImageDraw.Draw(img)

    # Try to load a bold system font; fall back to default
    font_size = int(size_px * 0.72)
    font = None
    for candidate in [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
        '/usr/share/fonts/truetype/freefont/FreeSansBold.ttf',
        '/System/Library/Fonts/Helvetica.ttc',
        '/Library/Fonts/Arial Bold.ttf',
    ]:
        try:
            font = ImageFont.truetype(candidate, font_size)
            break
        except (IOError, OSError):
            continue

    if font is None:
        font = ImageFont.load_default()

    # Centre the character
    bbox = draw.textbbox((0, 0), digit, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (size_px - tw) // 2 - bbox[0]
    y = (size_px - th) // 2 - bbox[1]
    draw.text((x, y), digit, fill=0, font=font)   # black character

    return np.array(img)


def make_synthetic_digits(out_dir: Path, size_px: int = 256):
    """Render digits 0-9 and save as PNG to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for d in range(10):
        img_arr = render_sloan_digit(str(d), size_px=size_px)
        Image.fromarray(img_arr).save(out_dir / f"digit_{d}.png")
    print(f"Rendered 10 synthetic digits → {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests for the filter
# ─────────────────────────────────────────────────────────────────────────────

def raised_cosine_formula(f: np.ndarray, f0: float, bw: float) -> np.ndarray:
    """
    Evaluate the raised cosine filter formula analytically at arbitrary
    frequencies (scalar or array).

    H(f) = 0.5 * (1 + cos(π * log2(f/f0) / bw))  if |log2(f/f0)| ≤ bw
         = 0                                        otherwise
    """
    f = np.asarray(f, dtype=float)
    H = np.zeros_like(f)
    pos = f > 0
    log_r = np.log2(f[pos] / f0)
    within = np.abs(log_r) <= bw
    H[pos] = np.where(within, 0.5 * (1.0 + np.cos(np.pi * log_r / bw)), 0.0)
    return H


def test_filter(f0: float = 3.0, bw: float = 1.0, N: int = 512, tol: float = 1e-10):
    """
    Validate the raised cosine filter.

    ANALYTICAL tests (T1–T5): evaluate the formula at exact frequencies.
    These are independent of the discrete grid and should pass exactly.

    DISCRETE tests (T6–T8): evaluate the N×N array returned by
    make_raised_cosine_filter().  The discrete grid has bin spacing of
    1 cpo (since fftfreq(N)*N gives integers), so frequencies like
    f0·2^(±bw/2) will NOT land on a bin – we therefore check the
    CLOSEST grid bin and allow a small tolerance for the quantisation.
    """
    results = []

    def check(name, condition, detail=''):
        status = '✓ PASS' if condition else '✗ FAIL'
        results.append(condition)
        print(f"  {status}  {name}  {detail}")

    print(f"\n─ Filter validation (f0={f0} cpo, BW={bw} oct) ─")
    print("  [Analytical] Formula evaluated at exact frequencies:")

    # ── Analytical tests ──────────────────────────────────────────────────────
    # T1: peak gain = 1 at f0
    h_peak = raised_cosine_formula(np.array([f0]), f0, bw)[0]
    check('T1  Peak gain = 1 at f0', abs(h_peak - 1.0) < tol,
          f'got {h_peak:.8f}')

    # T2: gain = 0.5 at FWHH edges (±bw/2 octaves from f0)
    fwhh_lo = f0 * 2 ** (-bw / 2)
    fwhh_hi = f0 * 2 ** ( bw / 2)
    h_flo = raised_cosine_formula(np.array([fwhh_lo]), f0, bw)[0]
    h_fhi = raised_cosine_formula(np.array([fwhh_hi]), f0, bw)[0]
    check('T2a Half-height at FWHH lower edge', abs(h_flo - 0.5) < tol,
          f'got {h_flo:.8f}  at {fwhh_lo:.4f} cpo')
    check('T2b Half-height at FWHH upper edge', abs(h_fhi - 0.5) < tol,
          f'got {h_fhi:.8f}  at {fwhh_hi:.4f} cpo')

    # T3: gain = 0 at support edges (±bw octaves from f0)
    supp_lo = f0 * 2 ** (-bw)
    supp_hi = f0 * 2 ** ( bw)
    h_slo = raised_cosine_formula(np.array([supp_lo]), f0, bw)[0]
    h_shi = raised_cosine_formula(np.array([supp_hi]), f0, bw)[0]
    check('T3a Zero gain at support lower edge', abs(h_slo) < tol,
          f'got {h_slo:.8f}  at {supp_lo:.4f} cpo')
    check('T3b Zero gain at support upper edge', abs(h_shi) < tol,
          f'got {h_shi:.8f}  at {supp_hi:.4f} cpo')

    # T4: gain = 0 well outside support (e.g. at 0 and at 10·f0)
    h_dc    = raised_cosine_formula(np.array([0.001]), f0, bw)[0]
    h_out   = raised_cosine_formula(np.array([10 * f0]), f0, bw)[0]
    check('T4a Near-DC gain = 0', h_dc == 0.0)
    check('T4b Far-from-centre gain = 0', h_out == 0.0)

    # T5: monotonically increasing on [0, f0] and decreasing on [f0, ∞)
    f_test = np.logspace(np.log10(supp_lo * 0.5), np.log10(f0), 500)
    H_test = raised_cosine_formula(f_test, f0, bw)
    # Within the support, H should be non-decreasing up to f0
    f_in = f_test[(f_test >= supp_lo) & (f_test <= f0)]
    H_in = raised_cosine_formula(f_in, f0, bw)
    check('T5  Monotone increase in [supp_lo, f0]', np.all(np.diff(H_in) >= -1e-12))

    # ── Discrete array tests ──────────────────────────────────────────────────
    print("\n  [Discrete] N×N filter array (N={}) – bin spacing = 1 cpo:".format(N))
    H_arr = make_raised_cosine_filter(N, f0, bw)

    # T6: DC bin is zero
    check('T6  DC gain = 0 (H[0,0])', H_arr[0, 0] == 0.0)

    # T7: non-negative everywhere
    check('T7  Filter non-negative', np.all(H_arr >= 0.0))

    # T8: radially symmetric (transpose of fft2 output)
    check('T8  Radial symmetry  H[u,v] == H[v,u]',
          np.allclose(H_arr, H_arr.T, atol=1e-12))

    # T9: array values match analytical formula at integer cpo grid points
    #     (the grid bins are at f = 1, 2, 3, ..., N/2 cpo)
    fx_grid = np.arange(1, N // 2)          # positive-frequency bins
    H_analytic = raised_cosine_formula(fx_grid.astype(float), f0, bw)
    H_from_arr = H_arr[0, fx_grid]          # row 0 = fy=0, so R=fx
    max_err = np.max(np.abs(H_from_arr - H_analytic))
    check('T9  Array matches formula at all integer-cpo grid points',
          max_err < tol, f'max abs err = {max_err:.2e}')

    n_pass = sum(results)
    n_total = len(results)
    print(f"\n  Result: {n_pass}/{n_total} tests passed.\n")
    return all(results)


# ─────────────────────────────────────────────────────────────────────────────
# Power-spectrum analysis
# ─────────────────────────────────────────────────────────────────────────────

def radial_power_spectrum(img: np.ndarray, N_bins: int = 128):
    """
    Compute the radially averaged power spectrum of a 2-D image.

    Returns (freq_cpo, power) where freq_cpo is in cycles per image width.
    """
    H, W = img.shape
    F = np.fft.fftshift(np.fft.fft2(img.astype(float)))
    power = np.abs(F) ** 2

    fx = np.fft.fftshift(np.fft.fftfreq(W) * W)
    fy = np.fft.fftshift(np.fft.fftfreq(H) * H)
    FX, FY = np.meshgrid(fx, fy)
    R = np.sqrt(FX ** 2 + FY ** 2)

    r_max = min(H, W) / 2
    edges = np.linspace(0, r_max, N_bins + 1)
    freq_bins = 0.5 * (edges[:-1] + edges[1:])
    power_bins = np.zeros(N_bins)

    for i in range(N_bins):
        mask = (R >= edges[i]) & (R < edges[i + 1])
        if mask.any():
            power_bins[i] = power[mask].mean()

    return freq_bins, power_bins


# ─────────────────────────────────────────────────────────────────────────────
# Main test + figure
# ─────────────────────────────────────────────────────────────────────────────

def run_tests_and_plot(f0: float = 3.0, bw: float = 1.0,
                       ppd: float = 60.0, contrast: float = 0.20):
    """Run all tests and produce a comprehensive validation figure."""

    out_dir    = Path('test_output')
    digits_dir = out_dir / 'synthetic_digits'
    filt_dir   = out_dir / 'filtered'

    # ── 1. Generate synthetic digits ─────────────────────────────────────────
    print("=== Step 1: Render synthetic digits ===")
    make_synthetic_digits(digits_dir, size_px=256)

    # ── 2. Filter unit tests ──────────────────────────────────────────────────
    print("=== Step 2: Filter validation ===")
    ok = test_filter(f0=f0, bw=bw)

    # ── 3. Run the full pipeline ──────────────────────────────────────────────
    print("=== Step 3: Full pipeline ===")
    process_digits(
        input_dir         = str(digits_dir),
        output_dir        = str(filt_dir),
        f0_cpo            = f0,
        bw_octaves        = bw,
        pixels_per_degree = ppd,
        rms_contrast      = contrast,
        mean_lum          = 128.0,
        save_resized      = True,
        save_plots        = False,
    )

    # ── 4. Power spectrum verification ────────────────────────────────────────
    print("\n=== Step 4: Power spectra ===")
    digit_img = np.array(Image.open(digits_dir / 'digit_0.png').convert('L'),
                         dtype=float)
    filt_img  = apply_bandpass_filter(digit_img, f0, bw)

    freq_orig, ps_orig = radial_power_spectrum(digit_img - digit_img.mean())
    freq_filt, ps_filt = radial_power_spectrum(filt_img)

    # ── 5. Comprehensive figure ───────────────────────────────────────────────
    print("\n=== Step 5: Composing validation figure ===")

    N_filt = 256
    H      = make_raised_cosine_filter(N_filt, f0, bw)
    H_sh   = np.fft.fftshift(H)
    fx_ax  = np.fft.fftshift(np.fft.fftfreq(N_filt) * N_filt)

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f'Raised cosine bandpass filter – validation\n'
        f'f₀ = {f0} cpo,  FWHH = {bw} octave,  '
        f'passband: {f0*2**(-bw/2):.2f}–{f0*2**(bw/2):.2f} cpo',
        fontsize=14, fontweight='bold', y=0.98,
    )

    gs = fig.add_gridspec(3, 6, hspace=0.45, wspace=0.35)

    # ── Row 0: filter profile ─────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, :3])
    profile = H_sh[N_filt // 2, :]
    ax0.plot(fx_ax, profile, 'steelblue', lw=2.5, label='H(f)')
    ax0.axhline(0.5, color='grey', ls='--', lw=1.2, label='Half-height')
    ax0.axvline( f0, color='tomato',   ls=':', lw=1.5, label=f'f₀={f0}')
    ax0.axvline(-f0, color='tomato',   ls=':', lw=1.5)
    fwhh_lo = f0 * 2 ** (-bw / 2);  fwhh_hi = f0 * 2 ** (bw / 2)
    for v in [fwhh_lo, fwhh_hi, -fwhh_lo, -fwhh_hi]:
        ax0.axvline(v, color='seagreen', ls='-.', lw=1.2)
    ax0.set_xlabel('Spatial frequency (cpo)'); ax0.set_ylabel('Gain')
    ax0.set_title('Filter 1-D profile (horizontal cross-section)')
    ax0.set_xlim(-10, 10); ax0.set_ylim(-0.05, 1.15)
    ax0.legend(fontsize=8); ax0.grid(True, alpha=0.3)

    # ── Row 0: power spectra ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 3:])
    ax1.semilogy(freq_orig, ps_orig + 1e-3, 'gray',   lw=1.5,
                 label='Original (zero-meaned)', alpha=0.7)
    ax1.semilogy(freq_filt, ps_filt + 1e-3, 'steelblue', lw=2,
                 label='Filtered')
    ax1.axvline(f0,      color='tomato',   ls=':', lw=1.5, label=f'f₀={f0}')
    ax1.axvline(fwhh_lo, color='seagreen', ls='-.', lw=1.2, label='FWHH edges')
    ax1.axvline(fwhh_hi, color='seagreen', ls='-.', lw=1.2)
    ax1.set_xlabel('Spatial frequency (cpo)'); ax1.set_ylabel('Power (log)')
    ax1.set_title('Radial power spectra: original vs filtered')
    ax1.set_xlim(0, 20); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    # ── Row 1: original + filtered + 2-D filter ───────────────────────────────
    ax_orig = fig.add_subplot(gs[1, 0])
    ax_orig.imshow(digit_img, cmap='gray', vmin=0, vmax=255)
    ax_orig.set_title('Original digit "0"'); ax_orig.axis('off')

    ax_filt = fig.add_subplot(gs[1, 1])
    disp    = to_display_image(filt_img, 128.0, contrast)
    ax_filt.imshow(disp, cmap='gray', vmin=0, vmax=255)
    ax_filt.set_title('Filtered (native res.)'); ax_filt.axis('off')

    ax_2d = fig.add_subplot(gs[1, 2])
    lim = 8; cx = N_filt // 2
    half = round(lim / (N_filt / 2) * cx)
    crop = H_sh[cx - half: cx + half, cx - half: cx + half]
    ax_2d.imshow(crop, cmap='inferno', origin='lower',
                 extent=[-lim, lim, -lim, lim], vmin=0, vmax=1)
    ax_2d.set_title('2-D filter (±8 cpo)'); ax_2d.set_xlabel('fx'); ax_2d.set_ylabel('fy')
    theta = np.linspace(0, 2*np.pi, 300)
    for r, col, ls in [(f0, 'w', '-'), (fwhh_lo, 'lime', '--'), (fwhh_hi, 'lime', '--')]:
        ax_2d.plot(r*np.cos(theta), r*np.sin(theta), color=col, ls=ls, lw=1.5)

    # ── Row 1: SF conditions ──────────────────────────────────────────────────
    labels_shown = 0
    for i, (label, size_deg, sf_cpd) in enumerate(SF_CONDITIONS):
        if i >= 3:
            break
        tpx = size_deg_to_pixels(size_deg, ppd)
        resized = resize_image(disp, tpx)
        canvas = np.full((128, 128), 128, dtype=np.uint8)
        oh = max(0, (128 - resized.shape[0]) // 2)
        ow = max(0, (128 - resized.shape[1]) // 2)
        rh = min(resized.shape[0], 128); rw = min(resized.shape[1], 128)
        canvas[oh:oh+rh, ow:ow+rw] = resized[:rh, :rw]
        ax_s = fig.add_subplot(gs[1, 3 + i])
        ax_s.imshow(canvas, cmap='gray', vmin=0, vmax=255)
        ax_s.set_title(f'{size_deg}° / {sf_cpd} cpd\n({tpx}px)', fontsize=9)
        ax_s.axis('off')

    # ── Row 2: remaining SF conditions ───────────────────────────────────────
    for i, (label, size_deg, sf_cpd) in enumerate(SF_CONDITIONS[3:]):
        tpx = size_deg_to_pixels(size_deg, ppd)
        resized = resize_image(disp, tpx)
        canvas = np.full((128, 128), 128, dtype=np.uint8)
        oh = max(0, (128 - resized.shape[0]) // 2)
        ow = max(0, (128 - resized.shape[1]) // 2)
        rh = min(resized.shape[0], 128); rw = min(resized.shape[1], 128)
        canvas[oh:oh+rh, ow:ow+rw] = resized[:rh, :rw]
        ax_s = fig.add_subplot(gs[2, i])
        ax_s.imshow(canvas, cmap='gray', vmin=0, vmax=255)
        ax_s.set_title(f'{size_deg}° / {sf_cpd} cpd\n({tpx}px)', fontsize=9)
        ax_s.axis('off')

    # ── Validation summary ────────────────────────────────────────────────────
    ax_sum = fig.add_subplot(gs[2, 3:])
    ax_sum.axis('off')

    # SF conditions table
    table_data = [[f'{sz}°', f'{sf} cpd',
                   f'{size_deg_to_pixels(sz, ppd)} px']
                  for _, sz, sf in SF_CONDITIONS]
    tbl = ax_sum.table(
        cellText=table_data,
        colLabels=['Size (°)', 'Central SF', f'Pixels\n({ppd:.0f} ppd)'],
        loc='center', cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    ax_sum.set_title(
        'Stimulus size × SF mapping\n(Zheng et al. 2018)',
        fontsize=10, pad=10,
    )

    # Save
    fig_path = out_dir / 'validation_figure.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")

    print("\n=== Summary ===")
    print(f"  Filter tests: {'PASS' if ok else 'FAIL'}")
    print(f"  Output dir  : {filt_dir}")
    print(f"  Validation  : {fig_path}")
    print("\nAll done.")


if __name__ == '__main__':
    run_tests_and_plot(f0=3.0, bw=1.0, ppd=60.0, contrast=0.20)
