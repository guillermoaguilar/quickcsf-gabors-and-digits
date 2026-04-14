#!/usr/bin/env python3
"""
test_generate_stimuli.py
========================
Self-contained test and validation suite for generate_stimuli.py.

What it does
------------
1. Renders synthetic Sloan-style digit images (0-9).
2. Validates the raised cosine filter analytically (12 unit tests).
3. Runs the full pipeline: one filtered image per digit per SF condition,
   all at the same pixel resolution, by varying f0.
4. Saves a validation figure:
   - All six filter profiles overlaid
   - Power spectra: original vs each SF condition for digit "0"
   - Image strip: original + all six filtered versions

Run:  python test_generate_stimuli.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from generate_stimuli import (
    SF_CONDITIONS,
    apply_bandpass_filter,
    f0_for_condition,
    make_raised_cosine_filter,
    next_power_of_two,
    process_digits,
    to_display_image,
)

# ---------------------------------------------------------------------------
# Synthetic digit renderer
# ---------------------------------------------------------------------------

def render_sloan_digit(digit: str, size_px: int = 256) -> np.ndarray:
    """Render a digit as black on mid-grey (128) background."""
    img  = Image.new('L', (size_px, size_px), color=128)
    draw = ImageDraw.Draw(img)
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

    bbox = draw.textbbox((0, 0), digit, font=font)
    x = (size_px - (bbox[2] - bbox[0])) // 2 - bbox[0]
    y = (size_px - (bbox[3] - bbox[1])) // 2 - bbox[1]
    draw.text((x, y), digit, fill=0, font=font)
    return np.array(img)


def make_synthetic_digits(out_dir: Path, size_px: int = 256):
    out_dir.mkdir(parents=True, exist_ok=True)
    for d in range(10):
        img_arr = render_sloan_digit(str(d), size_px=size_px)
        Image.fromarray(img_arr).save(out_dir / f"digit_{d}.png")
    print(f"Rendered 10 synthetic digits -> {out_dir}")


# ---------------------------------------------------------------------------
# Analytical formula (for unit tests)
# ---------------------------------------------------------------------------

def raised_cosine_formula(f: np.ndarray, f0: float, bw: float) -> np.ndarray:
    f  = np.asarray(f, dtype=float)
    H  = np.zeros_like(f)
    pos   = f > 0
    log_r = np.log2(f[pos] / f0)
    within = np.abs(log_r) <= bw
    H[pos] = np.where(within,
                      0.5 * (1.0 + np.cos(np.pi * log_r / bw)), 0.0)
    return H


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_filter(f0: float = 3.0, bw: float = 1.0,
                N: int = 512, tol: float = 1e-10) -> bool:
    """12 unit tests covering the analytical formula and the discrete array."""
    results = []

    def check(name, cond, detail=''):
        status = 'PASS' if cond else 'FAIL'
        results.append(cond)
        print(f"  {'✓' if cond else '✗'} {status}  {name}  {detail}")

    print(f"\n-- Filter validation (f0={f0} cpo, BW={bw} oct) --")
    print("  [Analytical]")

    # T1: peak
    h_peak = raised_cosine_formula(np.array([f0]), f0, bw)[0]
    check('T1  Peak gain = 1 at f0', abs(h_peak - 1.0) < tol,
          f'got {h_peak:.8f}')

    # T2: FWHH edges
    fwhh_lo = f0 * 2 ** (-bw / 2)
    fwhh_hi = f0 * 2 ** ( bw / 2)
    h_flo = raised_cosine_formula(np.array([fwhh_lo]), f0, bw)[0]
    h_fhi = raised_cosine_formula(np.array([fwhh_hi]), f0, bw)[0]
    check('T2a Half-height at FWHH lower edge',
          abs(h_flo - 0.5) < tol, f'got {h_flo:.8f}  at {fwhh_lo:.4f} cpo')
    check('T2b Half-height at FWHH upper edge',
          abs(h_fhi - 0.5) < tol, f'got {h_fhi:.8f}  at {fwhh_hi:.4f} cpo')

    # T3: support edges
    supp_lo = f0 * 2 ** (-bw)
    supp_hi = f0 * 2 ** ( bw)
    h_slo = raised_cosine_formula(np.array([supp_lo]), f0, bw)[0]
    h_shi = raised_cosine_formula(np.array([supp_hi]), f0, bw)[0]
    check('T3a Zero gain at support lower edge',
          abs(h_slo) < tol, f'got {h_slo:.8f}  at {supp_lo:.4f} cpo')
    check('T3b Zero gain at support upper edge',
          abs(h_shi) < tol, f'got {h_shi:.8f}  at {supp_hi:.4f} cpo')

    # T4: outside support
    h_near_dc = raised_cosine_formula(np.array([0.001]), f0, bw)[0]
    h_far     = raised_cosine_formula(np.array([10 * f0]), f0, bw)[0]
    check('T4a Near-DC gain = 0', h_near_dc == 0.0)
    check('T4b Far-from-centre gain = 0', h_far == 0.0)

    # T5: monotone increase on [supp_lo, f0]
    f_in = np.linspace(supp_lo, f0, 500)
    H_in = raised_cosine_formula(f_in, f0, bw)
    check('T5  Monotone increase in [supp_lo, f0]',
          np.all(np.diff(H_in) >= -1e-12))

    print("  [Discrete array]")
    H_arr = make_raised_cosine_filter(N, f0, bw)

    # T6: DC = 0
    check('T6  DC gain = 0 (H[0,0])', H_arr[0, 0] == 0.0)

    # T7: non-negative
    check('T7  Filter non-negative', np.all(H_arr >= 0.0))

    # T8: radial symmetry
    check('T8  Radial symmetry H[u,v] == H[v,u]',
          np.allclose(H_arr, H_arr.T, atol=1e-12))

    # T9: matches formula at integer-cpo grid points
    fx_grid    = np.arange(1, N // 2)
    H_analytic = raised_cosine_formula(fx_grid.astype(float), f0, bw)
    H_from_arr = H_arr[0, fx_grid]   # row 0 (fy=0) → R = fx
    max_err    = np.max(np.abs(H_from_arr - H_analytic))
    check('T9  Array matches formula at integer-cpo grid points',
          max_err < tol, f'max abs err = {max_err:.2e}')

    n_pass = sum(results)
    print(f"\n  Result: {n_pass}/{len(results)} tests passed.\n")
    return all(results)


# ---------------------------------------------------------------------------
# Power-spectrum helper
# ---------------------------------------------------------------------------

def radial_power_spectrum(img: np.ndarray, N_bins: int = 128):
    H, W = img.shape
    F     = np.fft.fftshift(np.fft.fft2(img.astype(float)))
    power = np.abs(F) ** 2
    fx = np.fft.fftshift(np.fft.fftfreq(W) * W)
    fy = np.fft.fftshift(np.fft.fftfreq(H) * H)
    FX, FY = np.meshgrid(fx, fy)
    R = np.sqrt(FX ** 2 + FY ** 2)
    r_max  = min(H, W) / 2
    edges  = np.linspace(0, r_max, N_bins + 1)
    freqs  = 0.5 * (edges[:-1] + edges[1:])
    powers = np.zeros(N_bins)
    for i in range(N_bins):
        mask = (R >= edges[i]) & (R < edges[i + 1])
        if mask.any():
            powers[i] = power[mask].mean()
    return freqs, powers


# ---------------------------------------------------------------------------
# Main: run tests + produce validation figure
# ---------------------------------------------------------------------------

def run_tests_and_plot(display_size_deg: float = 3.0,
                       bw: float = 1.0,
                       contrast: float = 0.20):

    out_dir    = Path('test_output')
    digits_dir = out_dir / 'synthetic_digits'
    filt_dir   = out_dir / 'filtered'

    # 1. Synthetic digits
    print("=== Step 1: Render synthetic digits ===")
    #make_synthetic_digits(digits_dir, size_px=256)

    # 2. Unit tests  (validate at f0=3, a nice round value)
    print("=== Step 2: Filter unit tests ===")
    ok = test_filter(f0=3.0, bw=bw)

    # 3. Full pipeline
    print("=== Step 3: Full pipeline ===")
    process_digits(
        input_dir        = str(digits_dir),
        output_dir       = str(filt_dir),
        display_size_deg = display_size_deg,
        bw_octaves       = bw,
        rms_contrast     = contrast,
        mean_lum         = 128.0,
        save_plots       = False,
    )

    # 4. Power spectra for digit "0"
    print("\n=== Step 4: Power spectra ===")
    digit0 = np.array(
        Image.open(digits_dir / 'digit_5.png').convert('L'), dtype=float)
    freq_orig, ps_orig = radial_power_spectrum(digit0 - digit0.mean())

    spectra = []
    for label, sf_cpd in SF_CONDITIONS:
        f0 = f0_for_condition(sf_cpd, display_size_deg)
        filt = apply_bandpass_filter(digit0, f0, bw)
        spectra.append((sf_cpd, *radial_power_spectrum(filt)))

    # 5. Validation figure
    print("\n=== Step 5: Composing validation figure ===")

    N_plot = 512
    cx     = N_plot // 2
    fx_ax  = np.fft.fftshift(np.fft.fftfreq(N_plot) * N_plot)
    colours = plt.cm.viridis(np.linspace(0.1, 0.9, len(SF_CONDITIONS)))

    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(
        f'Stimulus validation — display size = {display_size_deg} deg, '
        f'BW = {bw} oct, contrast = {contrast}',
        fontsize=13, fontweight='bold', y=0.99,
    )
    gs = fig.add_gridspec(3, len(SF_CONDITIONS) + 1,
                          hspace=0.5, wspace=0.35)

    # Row 0 left: filter profiles
    ax0 = fig.add_subplot(gs[0, :4])
    for (label, sf_cpd), col in zip(SF_CONDITIONS, colours):
        f0 = f0_for_condition(sf_cpd, display_size_deg)
        H  = make_raised_cosine_filter(N_plot, f0, bw)
        ax0.plot(fx_ax, np.fft.fftshift(H)[cx, :],
                 color=col, lw=2, label=f'{sf_cpd} cpd (f0={f0:.1f})')
    ax0.axhline(0.5, color='grey', ls='--', lw=1, alpha=0.6)
    ax0.set_xlabel('Spatial frequency (cpo)'); ax0.set_ylabel('Gain')
    ax0.set_title('Filter profiles for all SF conditions')
    f0_max = f0_for_condition(SF_CONDITIONS[-1][1], display_size_deg)
    ax0.set_xlim(-min(f0_max * 2.5, N_plot / 2),
                  min(f0_max * 2.5, N_plot / 2))
    ax0.set_ylim(-0.05, 1.15)
    ax0.legend(fontsize=7, ncol=2); ax0.grid(True, alpha=0.3)

    # Row 0 right: power spectra
    ax1 = fig.add_subplot(gs[0, 4:])
    ax1.semilogy(freq_orig, ps_orig + 1e-3,
                 color='grey', lw=1.5, label='Original', alpha=0.7)
    for (sf_cpd, freq_f, ps_f), col in zip(spectra, colours):
        ax1.semilogy(freq_f, ps_f + 1e-3,
                     color=col, lw=1.5, label=f'{sf_cpd} cpd')
    ax1.set_xlabel('Spatial frequency (cpo)'); ax1.set_ylabel('Power (log)')
    ax1.set_title('Radial power spectra')
    ax1.set_xlim(0, min(f0_max * 2, N_plot / 2))
    ax1.legend(fontsize=7, ncol=2); ax1.grid(True, alpha=0.3)

    # Row 1: original + first 3 SF conditions
    ax_orig = fig.add_subplot(gs[1, 0])
    ax_orig.imshow(digit0.astype(np.uint8), cmap='gray', vmin=0, vmax=255)
    ax_orig.set_title('Original\n(256 px)', fontsize=9); ax_orig.axis('off')

    for i, ((label, sf_cpd), col) in enumerate(
            zip(SF_CONDITIONS[:4], colours)):
        f0   = f0_for_condition(sf_cpd, display_size_deg)
        disp = to_display_image(apply_bandpass_filter(digit0, f0, bw),
                                128.0, contrast)
        ax_s = fig.add_subplot(gs[1, i + 1])
        ax_s.imshow(disp, cmap='gray', vmin=0, vmax=255)
        ax_s.set_title(f'{sf_cpd} cpd\n(f0={f0:.1f} cpo)', fontsize=9)
        ax_s.axis('off')

    # Row 2: remaining SF conditions + summary table
    remaining = list(zip(SF_CONDITIONS[4:], colours[4:]))
    for i, ((label, sf_cpd), col) in enumerate(remaining):
        f0   = f0_for_condition(sf_cpd, display_size_deg)
        disp = to_display_image(apply_bandpass_filter(digit0, f0, bw),
                                128.0, contrast)
        ax_s = fig.add_subplot(gs[2, i])
        ax_s.imshow(disp, cmap='gray', vmin=0, vmax=255)
        ax_s.set_title(f'{sf_cpd} cpd\n(f0={f0:.1f} cpo)', fontsize=9)
        ax_s.axis('off')

    # Summary table
    ax_tbl = fig.add_subplot(gs[2, len(remaining):])
    ax_tbl.axis('off')
    N_img   = next_power_of_two(256)
    nyquist = N_img / 2
    rows = []
    for label, sf_cpd in SF_CONDITIONS:
        f0      = f0_for_condition(sf_cpd, display_size_deg)
        supp_hi = f0 * 2 ** bw
        warn    = '!' if supp_hi > nyquist else ''
        rows.append([f'{sf_cpd}', f'{f0:.2f}', f'{supp_hi:.1f}{warn}'])
    tbl = ax_tbl.table(
        cellText=rows,
        colLabels=['SF (cpd)', 'f0 (cpo)', 'Supp.hi'],
        loc='center', cellLoc='center',
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.4)
    ax_tbl.set_title(f'Filter table\n(Nyquist={nyquist:.0f} cpo)',
                     fontsize=9, pad=6)

    fig_path = out_dir / 'validation_figure.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")

    print("\n=== Summary ===")
    print(f"  Filter tests : {'PASS' if ok else 'FAIL'}")
    print(f"  Output dir   : {filt_dir}")
    print(f"  Figure       : {fig_path}")
    print("\nAll done.")


if __name__ == '__main__':
    run_tests_and_plot(display_size_deg=5.0, bw=1.0, contrast=0.20)
