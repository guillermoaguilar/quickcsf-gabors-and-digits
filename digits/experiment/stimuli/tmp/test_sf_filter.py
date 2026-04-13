"""
Tests for sf_filter.py — raised cosine spatial frequency bandpass filters.

Run with:
    python test_sf_filter.py
or:
    pytest test_sf_filter.py -v

Test organisation
-----------------
1.  TestFilterShape              – output shape, value range, DC=0, no NaN
2.  TestCentreFrequency          – peak location, shift behaviour (cpo mode)
3.  TestBandwidth                – FBHH, zeros at edges, log-symmetry
4.  TestGratingSelectivity       – pass/reject gratings in cpo mode
5.  TestCentreFrequencyShift     – octave-shift property, bank consistency
6.  TestApplyFilter              – API: dc_offset, dtype, shape
7.  TestFilterBank               – batch API
8.  TestCpdMode                  – cpd ↔ cpo equivalence and independence
9.  TestUnitConversionHelpers    – cpo_to_cpd / cpd_to_cpo
10. TestEdgeCases                – invalid inputs, rectangular images
"""

import numpy as np
import pytest
from sf_filter import (
    make_raised_cosine_filter,
    apply_raised_cosine_filter,
    apply_filter_bank,
    cpo_to_cpd,
    cpd_to_cpo,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sine_grating(size: int, freq_cpo: float, angle_deg: float = 0.0) -> np.ndarray:
    """Sinusoidal grating at a given integer frequency and orientation."""
    x = np.linspace(0, 1, size, endpoint=False)
    X, Y = np.meshgrid(x, x)
    theta = np.deg2rad(angle_deg)
    phase = freq_cpo * (X * np.cos(theta) + Y * np.sin(theta))
    return np.sin(2 * np.pi * phase)


def _filter_value_at_freq(filt: np.ndarray, freq_cpo: float, n_eff: float) -> float:
    """
    Return the filter value at a given radial frequency by looking up the
    nearest pixel on the horizontal (fy=0) axis.

    NOTE: only reliable when ``freq_cpo`` is an integer, so the frequency
    falls on an exact FFT bin with no aliasing.
    """
    n = filt.shape[1]
    filt_s = np.fft.fftshift(filt)
    fx = np.fft.fftshift(np.fft.fftfreq(n)) * n_eff
    mid = filt.shape[0] // 2
    idx = np.argmin(np.abs(fx - freq_cpo))
    return float(filt_s[mid, idx])


def _interpolated_fbhh(filt: np.ndarray, n_eff: float) -> float:
    """
    Measure the full bandwidth at half-height (FBHH) in octaves by linearly
    interpolating the two 0.5-crossings on the horizontal frequency axis.
    """
    n = filt.shape[1]
    filt_s = np.fft.fftshift(filt)
    fx = np.fft.fftshift(np.fft.fftfreq(n)) * n_eff
    mid = filt.shape[0] // 2
    row = filt_s[mid, :]

    crossings = []
    for i in range(len(row) - 1):
        if fx[i] > 0 and (row[i] - 0.5) * (row[i + 1] - 0.5) < 0:
            t = (0.5 - row[i]) / (row[i + 1] - row[i])
            crossings.append(fx[i] + t * (fx[i + 1] - fx[i]))

    assert len(crossings) == 2, f"Expected 2 half-height crossings, found {len(crossings)}"
    return float(np.log2(crossings[1] / crossings[0]))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Filter shape and values
# ─────────────────────────────────────────────────────────────────────────────

class TestFilterShape:

    def test_output_shape_square(self):
        assert make_raised_cosine_filter(256, 256).shape == (256, 256)

    def test_output_shape_rectangular(self):
        assert make_raised_cosine_filter(128, 256).shape == (128, 256)

    def test_values_in_range(self):
        filt = make_raised_cosine_filter(256, 256)
        assert filt.min() >= -1e-12
        assert filt.max() <= 1.0 + 1e-12

    def test_dc_is_zero(self):
        filt = make_raised_cosine_filter(256, 256, center_freq=3.0)
        assert filt[0, 0] == pytest.approx(0.0, abs=1e-10)

    def test_peak_is_one(self):
        filt = make_raised_cosine_filter(256, 256, center_freq=3.0)
        assert filt.max() == pytest.approx(1.0, abs=1e-10)

    def test_filter_is_real(self):
        assert np.isrealobj(make_raised_cosine_filter(64, 64))

    def test_no_nan_or_inf(self):
        filt = make_raised_cosine_filter(64, 64, center_freq=3.0)
        assert np.all(np.isfinite(filt))

    def test_radially_symmetric_on_axes(self):
        """H(r,0) == H(0,r) for all r: the filter depends only on |f|."""
        n = 128
        filt_s = np.fft.fftshift(make_raised_cosine_filter(n, n, center_freq=3.0))
        mid = n // 2
        h_slice = filt_s[mid, mid:]
        v_slice = filt_s[mid:, mid]
        k = min(len(h_slice), len(v_slice))
        np.testing.assert_allclose(h_slice[:k], v_slice[:k], atol=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Centre frequency positioning (cpo mode)
# ─────────────────────────────────────────────────────────────────────────────

class TestCentreFrequency:

    @pytest.mark.parametrize("fc", [1.5, 3.0, 6.0, 12.0])
    def test_peak_near_centre_frequency(self, fc: float):
        n = 512
        filt = make_raised_cosine_filter(n, n, center_freq=fc)
        filt_s = np.fft.fftshift(filt)
        fy = np.fft.fftshift(np.fft.fftfreq(n)) * n
        fx = np.fft.fftshift(np.fft.fftfreq(n)) * n
        FX, FY = np.meshgrid(fx, fy)
        F = np.sqrt(FX**2 + FY**2)
        masked = filt_s.copy()
        masked[F < 0.5] = 0
        idx = np.unravel_index(np.argmax(masked), masked.shape)
        assert abs(F[idx] - fc) < 0.5

    def test_higher_centre_passes_higher_freq_better(self):
        n = 256
        g_lo = _sine_grating(n, freq_cpo=3.0)
        g_hi = _sine_grating(n, freq_cpo=12.0)
        f_lo = make_raised_cosine_filter(n, n, center_freq=3.0)
        f_hi = make_raised_cosine_filter(n, n, center_freq=12.0)

        def pwr(g, f):
            return float(np.sum(np.abs(np.fft.fft2(g) * f) ** 2))

        assert pwr(g_lo, f_lo) > pwr(g_hi, f_lo)
        assert pwr(g_hi, f_hi) > pwr(g_lo, f_hi)

    def test_default_centre_is_3_cpo(self):
        np.testing.assert_array_equal(
            make_raised_cosine_filter(128, 128),
            make_raised_cosine_filter(128, 128, center_freq=3.0),
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Bandwidth
# ─────────────────────────────────────────────────────────────────────────────

class TestBandwidth:

    @pytest.mark.parametrize("bw", [0.5, 1.0, 2.0])
    def test_fbhh_equals_bw_oct(self, bw: float):
        """
        FBHH measured via linear interpolation of the 0.5-crossing should
        match ``bw_oct`` within discrete-sampling tolerance (< 0.1 oct).
        """
        n = 512
        filt = make_raised_cosine_filter(n, n, center_freq=3.0, bw_oct=bw)
        fbhh = _interpolated_fbhh(filt, n_eff=float(n))
        assert abs(fbhh - bw) < 0.1, f"bw_oct={bw}: measured FBHH={fbhh:.4f}"

    def test_zero_at_bw_edges(self):
        """H should be 0 at ±bw_oct from centre (f_c/2 and f_c*2 for bw=1)."""
        n, fc = 512, 4.0
        filt = make_raised_cosine_filter(n, n, center_freq=fc, bw_oct=1.0)
        for f in (fc / 2, fc * 2):     # both are integers for fc=4
            val = _filter_value_at_freq(filt, f, n_eff=float(n))
            assert abs(val) < 0.05, f"H({f:.1f} cpo) = {val:.4f}, expected ≈0"

    def test_wider_bw_passes_more_energy(self):
        rng = np.random.default_rng(0)
        img = rng.standard_normal((128, 128))
        spectrum = np.fft.fft2(img)
        e_narrow = float(np.sum(np.abs(spectrum * make_raised_cosine_filter(128, 128, bw_oct=0.5)) ** 2))
        e_wide   = float(np.sum(np.abs(spectrum * make_raised_cosine_filter(128, 128, bw_oct=2.0)) ** 2))
        assert e_wide > e_narrow

    def test_bandwidth_log_symmetric_at_integer_pairs(self):
        """
        H(f_c·r) == H(f_c/r).  Use integer-cpo pairs where f₁×f₂ = fc²
        so both map to exact FFT bins (no interpolation needed).
        fc=6: pair (4,9) → 4×9=36=6², log₂(4/6)=−log₂(9/6).
        """
        n, fc = 512, 6.0
        filt = make_raised_cosine_filter(n, n, center_freq=fc, bw_oct=1.0)
        for f_lo, f_hi in [(4, 9), (3, 12)]:
            v_lo = _filter_value_at_freq(filt, f_lo, n_eff=float(n))
            v_hi = _filter_value_at_freq(filt, f_hi, n_eff=float(n))
            assert abs(v_lo - v_hi) < 0.01, (
                f"H({f_lo})={v_lo:.4f} vs H({f_hi})={v_hi:.4f}  (fc={fc})"
            )

    def test_peak_is_one_regardless_of_bw(self):
        for bw in [0.5, 1.0, 2.0]:
            filt = make_raised_cosine_filter(256, 256, center_freq=3.0, bw_oct=bw)
            assert filt.max() == pytest.approx(1.0, abs=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Grating selectivity (cpo mode, axis-aligned to avoid aliasing)
# ─────────────────────────────────────────────────────────────────────────────

class TestGratingSelectivity:

    @pytest.mark.parametrize("fc", [3.0, 6.0, 12.0])
    def test_passes_on_centre_grating(self, fc: float):
        """Horizontal grating at the filter centre should survive intact."""
        n = 256
        g = _sine_grating(n, freq_cpo=fc, angle_deg=0)
        out = apply_raised_cosine_filter(g, center_freq=fc, bw_oct=1.0, dc_offset=False)
        assert np.var(out) / np.var(g) > 0.90

    @pytest.mark.parametrize("fc,far", [
        (3.0, 24.0),   # 3 oct above fc=3  → H=0, exact integer bin
        (6.0,  1.0),   # ~2.6 oct below fc=6 → H=0, exact integer bin
    ])
    def test_rejects_far_off_grating(self, fc: float, far: float):
        """
        Grating at an integer cpo well outside the passband (so on an exact
        FFT bin, no spectral leakage) should be fully suppressed.
        """
        n = 256
        g = _sine_grating(n, freq_cpo=far, angle_deg=0)
        out = apply_raised_cosine_filter(g, center_freq=fc, bw_oct=1.0, dc_offset=False)
        assert np.var(out) / np.var(g) < 0.01

    def test_radially_symmetric_at_integer_cpo(self):
        """H(r,0) == H(0,r) for integer r (exact bin, no aliasing)."""
        n, fc = 256, 3.0
        filt_s = np.fft.fftshift(make_raised_cosine_filter(n, n, center_freq=fc))
        mid = n // 2
        for r in [1, 2, 3, 4, 5]:
            assert abs(filt_s[mid, mid + r] - filt_s[mid + r, mid]) < 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# 5. Centre-frequency shift properties
# ─────────────────────────────────────────────────────────────────────────────

class TestCentreFrequencyShift:

    def test_doubling_centre_shifts_one_octave_up(self):
        """H_2fc(f) == H_fc(f/2): doubling fc is a 1-octave log shift."""
        n = 512
        n_eff = float(n)
        f_lo = make_raised_cosine_filter(n, n, center_freq=3.0)
        f_hi = make_raised_cosine_filter(n, n, center_freq=6.0)
        for f in [3, 4, 6, 8, 10]:
            v_hi  = _filter_value_at_freq(f_hi, f,     n_eff)
            v_lo_h = _filter_value_at_freq(f_lo, f // 2, n_eff)  # f/2 integer for these values
            # compare only where f/2 is an integer (exact bin)
            if f % 2 == 0:
                assert abs(v_hi - v_lo_h) < 0.05, (
                    f"H_6({f})={v_hi:.3f}  H_3({f//2})={v_lo_h:.3f}"
                )

    def test_filters_at_different_centres_are_distinct(self):
        f1 = make_raised_cosine_filter(128, 128, center_freq=3.0)
        f2 = make_raised_cosine_filter(128, 128, center_freq=6.0)
        assert not np.allclose(f1, f2)

    @pytest.mark.parametrize("fc", [1.0, 2.0, 3.0, 6.0])
    def test_far_off_grating_rejected(self, fc: float):
        n = 256
        fc_far = fc * 8
        if fc_far > n / 4:
            pytest.skip("Far frequency too close to Nyquist")
        g = _sine_grating(n, freq_cpo=fc, angle_deg=0)
        out = apply_raised_cosine_filter(g, center_freq=fc_far, bw_oct=1.0, dc_offset=False)
        assert np.var(out) / np.var(g) < 0.01

    def test_shift_preserves_bandwidth(self):
        """Changing the centre must not change the filter bandwidth."""
        n = 512
        for fc in [3.0, 6.0]:
            filt = make_raised_cosine_filter(n, n, center_freq=fc, bw_oct=1.0)
            fbhh = _interpolated_fbhh(filt, n_eff=float(n))
            assert abs(fbhh - 1.0) < 0.1, f"fc={fc}: FBHH={fbhh:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# 6. apply_raised_cosine_filter API
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyFilter:

    def test_output_shape_preserved(self):
        img = np.random.default_rng(1).standard_normal((128, 128))
        assert apply_raised_cosine_filter(img).shape == (128, 128)

    def test_dc_offset_adds_half(self):
        rng = np.random.default_rng(2)
        img = rng.standard_normal((256, 256))
        img -= img.mean()
        out = apply_raised_cosine_filter(img, dc_offset=True)
        assert abs(out.mean() - 0.5) < 0.05

    def test_no_dc_offset_zero_mean(self):
        img = np.random.default_rng(3).standard_normal((256, 256))
        img -= img.mean()
        out = apply_raised_cosine_filter(img, dc_offset=False)
        assert abs(out.mean()) < 0.05

    def test_output_is_real(self):
        img = np.random.default_rng(4).standard_normal((64, 64))
        assert np.isrealobj(apply_raised_cosine_filter(img, dc_offset=False))

    def test_integer_input_handled(self):
        img = np.random.default_rng(5).integers(0, 256, size=(64, 64))
        out = apply_raised_cosine_filter(img)
        assert out.dtype == np.float64

    def test_filtering_reduces_power(self):
        img = np.random.default_rng(6).standard_normal((256, 256))
        out = apply_raised_cosine_filter(img, dc_offset=False)
        assert np.var(out) < np.var(img)


# ─────────────────────────────────────────────────────────────────────────────
# 7. apply_filter_bank
# ─────────────────────────────────────────────────────────────────────────────

class TestFilterBank:

    def test_returns_all_requested_centres(self):
        img = np.random.default_rng(7).standard_normal((64, 64))
        centres = [1.5, 3.0, 6.0, 12.0]
        results = apply_filter_bank(img, centers=centres)
        assert set(results.keys()) == set(centres)

    def test_matches_individual_calls(self):
        img = np.random.default_rng(8).standard_normal((64, 64))
        for fc in [3.0, 6.0]:
            bank = apply_filter_bank(img, centers=[fc])
            direct = apply_raised_cosine_filter(img, center_freq=fc)
            np.testing.assert_array_equal(bank[fc], direct)

    def test_empty_centres_list(self):
        assert apply_filter_bank(np.ones((32, 32)), centers=[]) == {}

    def test_cpd_bank_matches_individual_cpd_calls(self):
        """filter_bank in cpd mode must match individual apply calls."""
        rng = np.random.default_rng(9)
        img = rng.standard_normal((128, 128))
        ppd = 60.0
        centres_cpd = [2.0, 4.0, 8.0]
        bank = apply_filter_bank(img, centers=centres_cpd, ppd=ppd)
        for fc in centres_cpd:
            direct = apply_raised_cosine_filter(img, center_freq=fc, ppd=ppd)
            np.testing.assert_array_equal(bank[fc], direct)


# ─────────────────────────────────────────────────────────────────────────────
# 8. CPD mode
# ─────────────────────────────────────────────────────────────────────────────

class TestCpdMode:
    """
    Tests for the ppd-based (cycles per degree) interface.

    Core principle: cpd mode and cpo mode must produce identical filters when
    the implied centre frequency in cycles/pixel is the same.

    Conversion:
        f_cpp (cpo mode) = center_cpo  / n_eff
        f_cpp (cpd mode) = center_cpd  / ppd
    So the filters are identical when:
        center_cpo / n_eff  =  center_cpd / ppd
        center_cpd = center_cpo * ppd / n_eff
    """

    def _equivalent_cpd(self, center_cpo: float, n: int, ppd: float) -> float:
        """Return the cpd value that equals center_cpo for an n×n image at ppd."""
        return center_cpo * ppd / n

    def test_cpd_and_cpo_produce_identical_filter(self):
        """When center_cpd = center_cpo * ppd / n_eff, both filters must match."""
        n, ppd = 256, 60.0
        center_cpo = 3.0
        center_cpd = self._equivalent_cpd(center_cpo, n, ppd)

        filt_cpo = make_raised_cosine_filter(n, n, center_freq=center_cpo)
        filt_cpd = make_raised_cosine_filter(n, n, center_freq=center_cpd, ppd=ppd)

        np.testing.assert_allclose(filt_cpo, filt_cpd, atol=1e-14,
                                   err_msg="cpo and cpd filters differ despite same f_cpp")

    def test_cpd_filter_peak_near_correct_frequency(self):
        """
        For a known display (n=512 px, ppd=60), a 4-cpd filter should have
        its peak at the pixel corresponding to 4 cycles per degree.
        """
        n, ppd, fc_cpd = 512, 60.0, 4.0
        filt = make_raised_cosine_filter(n, n, center_freq=fc_cpd, ppd=ppd)
        filt_s = np.fft.fftshift(filt)
        # Frequency axis in cpd: f_cpp * ppd
        fx_cpd = np.fft.fftshift(np.fft.fftfreq(n)) * ppd
        fy_cpd = np.fft.fftshift(np.fft.fftfreq(n)) * ppd
        FX, FY = np.meshgrid(fx_cpd, fy_cpd)
        F_cpd = np.sqrt(FX**2 + FY**2)
        masked = filt_s.copy()
        masked[F_cpd < 0.5] = 0
        idx = np.unravel_index(np.argmax(masked), masked.shape)
        assert abs(F_cpd[idx] - fc_cpd) < 0.5

    def test_cpd_mode_dc_is_zero(self):
        filt = make_raised_cosine_filter(256, 256, center_freq=4.0, ppd=60.0)
        assert filt[0, 0] == pytest.approx(0.0, abs=1e-10)

    def test_cpd_mode_peak_is_one(self):
        """
        The filter peak should be at or very close to 1.0.

        In cpd mode, f_c_cpp = center_cpd / ppd, which generally does not fall
        on an exact FFT bin (unlike integer cpo values).  The nearest bin may
        be slightly off-centre, so the numerical peak can be marginally below 1.
        A tolerance of 1e-3 is used here; it is not a loose bound — it reflects
        the inherent ≤0.5-bin discretisation error of a Cartesian FFT grid.
        """
        filt = make_raised_cosine_filter(256, 256, center_freq=4.0, ppd=60.0)
        assert filt.max() == pytest.approx(1.0, abs=1e-3)

    def test_cpd_bandwidth_matches_bw_oct(self):
        """
        The FBHH of a cpd-mode filter measured in cpd units should match
        bw_oct (the bandwidth is unit-independent).
        """
        n, ppd = 512, 60.0
        filt = make_raised_cosine_filter(n, n, center_freq=4.0, ppd=ppd, bw_oct=1.0)
        # Measure bandwidth: use ppd as n_eff to work in cpd units
        fbhh = _interpolated_fbhh(filt, n_eff=ppd)
        assert abs(fbhh - 1.0) < 0.1, f"FBHH in cpd mode = {fbhh:.4f} oct"

    def test_different_ppd_gives_different_filter(self):
        """Two different ppd values must produce different filters."""
        f1 = make_raised_cosine_filter(256, 256, center_freq=4.0, ppd=60.0)
        f2 = make_raised_cosine_filter(256, 256, center_freq=4.0, ppd=30.0)
        assert not np.allclose(f1, f2)

    def test_higher_ppd_shifts_filter_to_higher_pixel_frequency(self):
        """
        Higher ppd (finer display) means more pixels per degree, so the same
        cpd value corresponds to a higher spatial frequency in cpp.  A grating
        at fc_cpd * ppd_pixels/image cpo should pass better for the correct ppd.
        """
        n = 256
        fc_cpd = 2.0
        for ppd in [30.0, 60.0]:
            fc_cpo_equiv = fc_cpd * n / ppd
            g = _sine_grating(n, freq_cpo=round(fc_cpo_equiv))
            out = apply_raised_cosine_filter(g, center_freq=fc_cpd, ppd=ppd,
                                             bw_oct=1.0, dc_offset=False)
            ratio = np.var(out) / np.var(g)
            # The grating is near the filter centre, so ratio should be high
            assert ratio > 0.5, f"ppd={ppd}: power ratio = {ratio:.3f}"

    def test_apply_filter_cpd_mode(self):
        """apply_raised_cosine_filter in cpd mode returns correct shape/dtype."""
        img = np.random.default_rng(10).standard_normal((128, 128))
        out = apply_raised_cosine_filter(img, center_freq=4.0, ppd=60.0)
        assert out.shape == img.shape
        assert out.dtype == np.float64

    def test_apply_filter_cpd_equals_cpo_equivalent(self):
        """
        apply_raised_cosine_filter must return identical results when called
        in cpd and in the equivalent cpo mode.
        """
        rng = np.random.default_rng(11)
        img = rng.standard_normal((256, 256))
        n, ppd = 256, 60.0
        center_cpo = 3.0
        center_cpd = center_cpo * ppd / n

        out_cpo = apply_raised_cosine_filter(img, center_freq=center_cpo, dc_offset=False)
        out_cpd = apply_raised_cosine_filter(img, center_freq=center_cpd, ppd=ppd,
                                             dc_offset=False)
        np.testing.assert_allclose(out_cpo, out_cpd, atol=1e-12)

    def test_no_ppd_and_ppd_none_are_identical(self):
        """Explicit ppd=None must behave identically to not passing ppd at all."""
        filt_implicit = make_raised_cosine_filter(128, 128, center_freq=3.0)
        filt_explicit = make_raised_cosine_filter(128, 128, center_freq=3.0, ppd=None)
        np.testing.assert_array_equal(filt_implicit, filt_explicit)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Unit-conversion helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestUnitConversionHelpers:

    def test_cpo_to_cpd_basic(self):
        """3 cpo in a 60-ppd, 180-px image → 1 cpd (3 / (180/60) = 3/3 = 1)."""
        assert cpo_to_cpd(3.0, image_size_px=180, ppd=60.0) == pytest.approx(1.0)

    def test_cpd_to_cpo_basic(self):
        """1 cpd in a 60-ppd, 180-px image → 3 cpo (1 * 3deg = 3)."""
        assert cpd_to_cpo(1.0, image_size_px=180, ppd=60.0) == pytest.approx(3.0)

    def test_round_trip_cpo_to_cpd_to_cpo(self):
        """cpo → cpd → cpo should recover the original value."""
        n, ppd, orig = 512, 42.5, 7.3
        assert cpd_to_cpo(cpo_to_cpd(orig, n, ppd), n, ppd) == pytest.approx(orig)

    def test_round_trip_cpd_to_cpo_to_cpd(self):
        n, ppd, orig = 256, 60.0, 4.0
        assert cpo_to_cpd(cpd_to_cpo(orig, n, ppd), n, ppd) == pytest.approx(orig)

    def test_larger_ppd_gives_smaller_cpo_for_same_cpd(self):
        """Same cpd on a higher-ppd display → smaller image-size in degrees → fewer cpo."""
        n = 256
        cpo_60 = cpd_to_cpo(4.0, n, ppd=60.0)
        cpo_30 = cpd_to_cpo(4.0, n, ppd=30.0)
        # At 30 ppd the image covers more degrees, so more cpo for same cpd
        assert cpo_30 > cpo_60

    def test_linearity(self):
        """Conversion is linear: 2× cpd → 2× cpo."""
        n, ppd = 256, 60.0
        assert cpd_to_cpo(8.0, n, ppd) == pytest.approx(2 * cpd_to_cpo(4.0, n, ppd))


# ─────────────────────────────────────────────────────────────────────────────
# 10. Edge cases and error handling
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_zero_centre_raises(self):
        with pytest.raises(ValueError):
            make_raised_cosine_filter(128, 128, center_freq=0)

    def test_negative_centre_raises(self):
        with pytest.raises(ValueError):
            make_raised_cosine_filter(128, 128, center_freq=-1.0)

    def test_zero_bandwidth_raises(self):
        with pytest.raises(ValueError):
            make_raised_cosine_filter(128, 128, bw_oct=0)

    def test_zero_ppd_raises(self):
        with pytest.raises(ValueError):
            make_raised_cosine_filter(128, 128, center_freq=4.0, ppd=0)

    def test_negative_ppd_raises(self):
        with pytest.raises(ValueError):
            make_raised_cosine_filter(128, 128, center_freq=4.0, ppd=-60.0)

    def test_1x1_image_does_not_crash(self):
        filt = make_raised_cosine_filter(1, 1, center_freq=3.0)
        assert filt.shape == (1, 1)

    def test_non_square_image(self):
        img = np.random.default_rng(12).standard_normal((128, 256))
        out = apply_raised_cosine_filter(img, center_freq=3.0)
        assert out.shape == (128, 256)

    def test_rectangular_uses_geometric_mean(self):
        h, w, fc = 128, 256, 3.0
        filt = make_raised_cosine_filter(h, w, center_freq=fc)
        filt_s = np.fft.fftshift(filt)
        n_eff = np.sqrt(h * w)
        fy = np.fft.fftshift(np.fft.fftfreq(h)) * n_eff
        fx = np.fft.fftshift(np.fft.fftfreq(w)) * n_eff
        FX, FY = np.meshgrid(fx, fy)
        F = np.sqrt(FX**2 + FY**2)
        masked = filt_s.copy()
        masked[F < 0.5] = 0
        idx = np.unravel_index(np.argmax(masked), masked.shape)
        assert abs(F[idx] - fc) < 1.0

    def test_no_nan_in_output(self):
        img = np.random.default_rng(13).standard_normal((64, 64))
        assert np.all(np.isfinite(apply_raised_cosine_filter(img)))

    def test_no_nan_in_output_cpd_mode(self):
        img = np.random.default_rng(14).standard_normal((64, 64))
        assert np.all(np.isfinite(apply_raised_cosine_filter(img, center_freq=4.0, ppd=60.0)))


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
