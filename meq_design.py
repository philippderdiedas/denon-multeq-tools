#!/usr/bin/env python3
"""MultEQ XT filter design in pure Python/numpy, reproducing the Audyssey MultEQ Editor app's calculation.

Input is an .ady file (impulse responses per channel and mic position); output per channel is a
512-tap minimum-phase FIR for the "reference" (Audyssey target) and "flat" curves at 32/44.1/48 kHz,
plus speaker cutoff, distance and level, in the same structure as meq_filter.compute_filters.
Tables (target curves, ACM1H mic correction) are loaded from meq_data/ next to this file.
"""
import os

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meq_data")

FS = 48000                 # analysis sample rate
FFT_LEN = 16384            # response length / FFT size
BINS = FFT_LEN // 2 + 1    # 8193
MIN_MAG = 1e-7

# Level weighting: C-weighting x pink, band-limited by 3rd-order Butterworth high/low pass, unit RMS
C_WEIGHT_A2_PLUS_B2 = 148840432.0   # 20.6^2 + 12200^2
C_WEIGHT_B2 = 148840000.0           # 12200^2
C_WEIGHT_A2B2 = 63161741312.0       # 20.6^2 * 12200^2
SAT_WEIGHT_BAND = (500.0, 2000.0)
SUB_WEIGHT_BAND = (30.0, 80.0)
WEIGHT_ORDER = 3


def _load(name):
    return np.load(os.path.join(DATA_DIR, name + ".npy")).astype(np.float64)


def fft_freqs(fs=FS, n=FFT_LEN):
    return np.arange(n // 2 + 1) * (fs / n)


def frequency_to_index(f, bins=BINS, fs=FS):
    return int(((bins * 2 - 2) * f) / fs + 0.5)


def butterworth_highpass(freqs, fc, order):
    y = (freqs / fc) ** order
    return y / np.sqrt(1.0 + y * y)


def butterworth_lowpass(freqs, fc, order, gain=1.0):
    y = (freqs / fc) ** order
    return gain / np.sqrt(1.0 + y * y)


def rms(x):
    return np.sqrt(np.mean(np.square(x)))


def level_weighting(freqs, band):
    """C-weighting x pink (1/sqrt f) x Butterworth band, normalized to unit RMS over all bins."""
    f2 = freqs * freqs
    c = C_WEIGHT_B2 / (f2 * (f2 + C_WEIGHT_A2_PLUS_B2) + C_WEIGHT_A2B2) * f2
    pink = np.empty_like(freqs)
    pink[1:] = 1.0 / np.sqrt(freqs[1:])
    pink[0] = pink[1]
    w = c * pink * butterworth_highpass(freqs, band[0], WEIGHT_ORDER) * butterworth_lowpass(freqs, band[1], WEIGHT_ORDER)
    return w / rms(w)


def mic_spectrum(mic_ir):
    """Complex spectrum of the mic correction impulse response (zero-padded to FFT_LEN)."""
    return np.fft.rfft(mic_ir, FFT_LEN)


def response_magnitude(ir, mic_spec):
    """|FFT(ir) x mic spectrum| over BINS, floored at MIN_MAG."""
    x = np.zeros(FFT_LEN)
    ir = np.asarray(ir, dtype=np.float64)[:FFT_LEN]
    x[:len(ir)] = ir
    return np.maximum(np.abs(np.fft.rfft(x) * mic_spec), MIN_MAG)


def weighted_level(mag, weighting):
    return rms(mag * weighting)


def average_response(irs, mic_spec, weighting):
    """Average of the level-normalized position magnitudes, renormalized to unit weighted level."""
    acc = np.zeros(BINS)
    for ir in irs:
        mag = response_magnitude(ir, mic_spec)
        acc += mag / weighted_level(mag, weighting)
    return acc / weighted_level(acc, weighting)


CUTOFF_TEST_FREQS = (20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0,
                     160.0, 170.0, 180.0, 190.0, 200.0, 250.0)
CUTOFF_FIT_RANGE = (11.72, 180.0)
CUTOFF_LEVEL_BAND = {False: (500.0, 2000.0), True: (46.875, 64.453)}
CUTOFF_ORDERS = range(2, 7)
CUTOFF_DEFAULT = 80.0


def detect_cutoff(avg, is_sub, freqs=None):
    """Speaker low-frequency cutoff: best-fitting Butterworth high pass (order 2..6) to the averaged response.

    The model is scaled to the response's RMS in a reference band and compared by mean |ln(model/response)|
    over 11.72..180 Hz. Satellites test 20..250 Hz, subwoofers 20..50 Hz. Returns (cutoff_hz, order).
    """
    freqs = fft_freqs() if freqs is None else freqs
    lo = frequency_to_index(CUTOFF_LEVEL_BAND[is_sub][0])
    hi = frequency_to_index(CUTOFF_LEVEL_BAND[is_sub][1])
    band = slice(lo, hi + 1)
    band_rms = rms(avg[band])
    f0, f1 = frequency_to_index(CUTOFF_FIT_RANGE[0]), frequency_to_index(CUTOFF_FIT_RANGE[1])
    fit = slice(f0, f1 + 1)
    tests = CUTOFF_TEST_FREQS[:4] if is_sub else CUTOFF_TEST_FREQS
    best, best_idx, best_order = np.inf, -1, 2
    for order in CUTOFF_ORDERS:
        for idx, fc in enumerate(tests):
            model = butterworth_highpass(freqs[fit], fc, order)
            if is_sub:
                model = model * (band_rms / rms(butterworth_highpass(freqs[band], fc, order)))
            else:
                model = model * band_rms
            err = np.mean(np.abs(np.log(model / avg[fit])))
            if err < best:
                best, best_idx, best_order = err, idx, order
    return (tests[best_idx] if best_idx >= 0 else CUTOFF_DEFAULT), best_order


# --- de Boor cubic smoothing spline on a soft-log frequency axis ----------------------------------
# x = ln(e^-4 + (e^4 - e^-4) * f / f_max): linear near 0 Hz, logarithmic above, spanning [-4, 4].

SPLINE_X_OFFSET = float(np.exp(-4.0))
SPLINE_X_TOP = float(np.exp(4.0))
SMOOTH_SAT = 0.99999        # EQ smoothing, satellites
SMOOTH_SUB = 0.9999995      # EQ smoothing, subwoofers; also used for resampling to 32/44.1 kHz


def spline_axis(freqs, f_max):
    return np.log(freqs * ((SPLINE_X_TOP - SPLINE_X_OFFSET) / f_max) + SPLINE_X_OFFSET)


def smoothing_spline(x, y, p):
    """de Boor's SMOOTH with unit weights and fixed p: minimizes p*sum((y-f)^2) + (1-p)*int(f''^2).

    Returns piecewise-polynomial coefficients (a, c1, c2, c3) per node: f = a + h*(c1 + h*(c2/2 + h*c3/6)).
    """
    n = len(x)
    h = np.diff(x)
    r = 1.0 / h
    six1mp = 6.0 * (1.0 - p)
    # Q^T Q (pentadiagonal, symmetric) and R (tridiagonal) on the interior nodes
    qtq0 = r[:-1] ** 2 + (r[:-1] + r[1:]) ** 2 + r[1:] ** 2
    qtq1 = -(r[1:-1] * (r[:-2] + r[1:-1]) + r[1:-1] * (r[1:-1] + r[2:]))
    qtq2 = r[1:-2] * r[2:-1]
    d0 = six1mp * qtq0 + 2.0 * p * (h[:-1] + h[1:])
    d1 = six1mp * qtq1 + p * h[1:-1]
    d2 = six1mp * qtq2
    qty = (y[2:] - y[1:-1]) * r[1:] - (y[1:-1] - y[:-2]) * r[:-1]
    u = _solve_pentadiagonal(d0, d1, d2, qty)
    uf = np.zeros(n)
    uf[1:-1] = u
    du = np.diff(uf) * r
    qu = np.zeros(n)
    qu[:-1] += du
    qu[1:] -= du
    a = y - six1mp * qu
    c2 = 6.0 * p * uf
    c3 = np.diff(c2) * r
    c1 = np.diff(a) * r - (c2[:-1] + c3 * h / 3.0) * 0.5 * h
    return a, c1, c2[:-1], c3


def _solve_pentadiagonal(d0, d1, d2, b):
    """Solve a symmetric positive definite pentadiagonal system via LDL^T (as in de Boor's CHOL1D)."""
    m = len(d0)
    d = d0.astype(np.float64).copy()
    l1 = np.zeros(m)
    l2 = np.zeros(m)
    e1 = np.append(d1, 0.0)
    e2 = np.append(d2, [0.0, 0.0])
    for i in range(m):
        if i >= 1:
            d[i] -= l1[i - 1] ** 2 * d[i - 1]
        if i >= 2:
            d[i] -= l2[i - 2] ** 2 * d[i - 2]
        if i < m - 1:
            v = e1[i]
            if i >= 1:
                v -= l1[i - 1] * l2[i - 1] * d[i - 1]
            l1[i] = v / d[i]
        if i < m - 2:
            l2[i] = e2[i] / d[i]
    z = b.astype(np.float64).copy()
    for i in range(1, m):
        z[i] -= l1[i - 1] * z[i - 1]
        if i >= 2:
            z[i] -= l2[i - 2] * z[i - 2]
    z /= d
    for i in range(m - 2, -1, -1):
        z[i] -= l1[i] * z[i + 1]
        if i + 2 < m:
            z[i] -= l2[i] * z[i + 2]
    return z


def eval_spline(x, coefs, xq):
    a, c1, c2, c3 = coefs
    i = np.clip(np.searchsorted(x, xq, side="right") - 1, 0, len(x) - 2)
    hq = xq - x[i]
    return a[i] + hq * (c1[i] + hq * (c2[i] * 0.5 + hq * c3[i] / 6.0))


def smooth_log(values, p, fs=FS, dst_fs=None):
    """Smooth (and optionally resample to 0..dst_fs/2) a curve sampled on the 0..fs/2 grid."""
    n = len(values)
    f = np.linspace(0.0, fs * 0.5, n)
    x = spline_axis(f, fs * 0.5)
    coefs = smoothing_spline(x, np.asarray(values, dtype=np.float64), p)
    xq = x if dst_fs is None else spline_axis(np.linspace(0.0, dst_fs * 0.5, n), fs * 0.5)
    return eval_spline(x, coefs, xq)


# --- interpolation --------------------------------------------------------------------------------

def interp_bezier(xp, yp, x):
    """Piecewise quadratic through (xp, yp) with zero slope at every point: each segment is split at its
    midpoint into two parabolas (ease-out to the midpoint value, then ease-in to the next point).
    Constant extension outside [xp[0], xp[-1]]."""
    xp = np.asarray(xp, dtype=np.float64)
    yp = np.asarray(yp, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    out[x < xp[0]] = yp[0]
    out[x >= xp[0]] = yp[-1]
    for k in range(1, len(xp)):
        x0, x1, y0, y1 = xp[k - 1], xp[k], yp[k - 1], yp[k]
        if k == 1:
            sel = (x >= x0) & (x <= x1)
        else:
            sel = (x > x0) & (x <= x1)
        xm, ym = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        xs = x[sel]
        first = xs < xm
        t1 = (xs - x0) / (xm - x0)
        t2 = (xs - xm) / (x1 - xm)
        out[sel] = np.where(first, y0 + (ym - y0) * t1 * t1, ym + t2 * (2.0 * (y1 - ym) + (ym - y1) * t2))
    return out


def interp_linear(xp, yp, x):
    return np.interp(x, xp, yp)


# --- minimum-phase FIR via linear prediction ------------------------------------------------------

def levinson_durbin(r, order, reg=1.0):
    """Prediction-error filter [1, -phi_1, ..., -phi_(order-1)] for autocorrelation r.

    Reflection coefficients are scaled by reg; returns None when the recursion becomes unstable
    (|forward error| >= prediction error power)."""
    phi = np.zeros(order - 1)
    err = r[0]
    for m in range(order - 1):
        acc = r[m + 1] - np.dot(phi[:m], r[m:0:-1]) if m else r[1]
        if abs(acc) >= err:
            return None
        k = acc * reg / err
        prev = phi[:m][::-1].copy()
        phi[m] = k
        err -= k * acc
        phi[:m] -= k * prev
    return np.concatenate(([1.0], -phi))


# --- target curve ---------------------------------------------------------------------------------

BASE_CURVES = {0: None, 1: "target_hfro1", 2: "target_hfro2", 3: "target_smpte202m"}
TARGET_HP_ORDER = 2
TARGET_FLOOR = 1e-5
MAX_BOOST_DB = 9.0
CUT_LIMIT_DB = -20.0
LEVEL_TOLERANCE_DB = 0.01
LEVEL_ITERATIONS = 51
_tables = {}


def table(name):
    if name not in _tables:
        _tables[name] = _load(name)
    return _tables[name]


def custom_curve(points, freqs):
    """Custom target points [(Hz, dB), ...] -> linear gain, interpolated in ln(f) with interp_bezier."""
    if not points:
        return np.ones_like(freqs)
    with np.errstate(divide="ignore"):
        lnf = np.log(freqs)
    db = interp_bezier(np.log([p[0] for p in points]), [p[1] for p in points], lnf)
    return 10.0 ** (db / 20.0)


def target_curve(freqs, points, base_curve, midrange, cutoff_option, effective_cutoff, dolby=False):
    """Target magnitude: custom curve x base curve (HFRO/SMPTE table) x midrange compensation
    x Dolby elevation x 2nd-order Butterworth high pass at the speaker cutoff (small speakers only)."""
    t = custom_curve(points, freqs)
    tf = table("target_frequencies")
    if BASE_CURVES.get(base_curve):
        t = t * interp_linear(tf, table(BASE_CURVES[base_curve]), freqs)
    if midrange:
        t = t * interp_linear(tf, table("target_midrange_comp"), freqs)
    if dolby:
        t = t * interp_linear(tf, table("target_dolby_elevation"), freqs)
    if cutoff_option > 0.0:
        t = t * butterworth_highpass(freqs, effective_cutoff, TARGET_HP_ORDER)
    return t + TARGET_FLOOR


def hf_limit_start(f):
    """Frequency where the EQ starts fading out towards the upper limit f."""
    if f < 20000.0:
        return 0.5 * f
    return min(f, 24000.0) * 2.5 - 40000.0


def eq_limits(freqs, effective_cutoff, cutoff_option, max_boost=MAX_BOOST_DB, cut_limit=CUT_LIMIT_DB):
    """Upper (boost) and lower (cut) gain limits, linear. Boost fades in from fc/2 to ~2.8*fc (log-f
    smoothstep), cut from 10 Hz to 1.5*fc + 10 Hz. Large speakers use the cutoff lowered by 10 Hz."""
    fc = effective_cutoff + (-10.0 if cutoff_option == 0.0 else 0.0)
    fc = max(fc, 10.0)
    k = (fc - 10.0) / 20.0 + 1.2 if fc < 30.0 else 2.0
    with np.errstate(divide="ignore"):
        lnf = np.log(freqs)
    lnf[0] = 1.0
    hi_x = [np.log(0.5 * fc), np.log(k * fc * np.sqrt(2.0) * max_boost / 9.0)]
    lo_x = [np.log(10.0), np.log(fc * 1.5 + 10.0)]
    hi = interp_bezier(hi_x, [0.0, max_boost], lnf)
    lo = interp_bezier(lo_x, [0.0, cut_limit], lnf)
    return 10.0 ** (hi / 20.0), 10.0 ** (lo / 20.0)


def level_eq(eq0, pos_resp, hi, lo, taper, hf_idx, weighting, target=1.0):
    """Scale the EQ so the corrected reference position has the target weighted level, clipping to the
    limits and fading out above the HF limit; up to 51 iterations, stop within 0.01 dB."""
    cur = weighted_level(eq0 * pos_resp, weighting)
    eq = eq0
    for _ in range(LEVEL_ITERATIONS):
        eq = np.clip(eq0 * (target / cur), lo, hi)
        eq[hf_idx:] = eq[hf_idx:] ** taper[hf_idx:]
        ratio = weighted_level(eq * pos_resp, weighting) / target
        cur *= ratio
        if abs(20.0 * np.log10(ratio)) <= LEVEL_TOLERANCE_DB:
            break
    return eq


def lpc_filter(eq, taps, fs):
    """Minimum-phase FIR whose magnitude follows eq (given on the 0..24 kHz grid), designed for rate fs:
    resample to 0..fs/2, LPC on the autocorrelation of 1/|eq|^2, normalize to unit gain at fs/2.
    Returns (coefficients, peak magnitude)."""
    if fs != FS:
        eq = np.exp(smooth_log(np.log(eq), SMOOTH_SUB, dst_fs=fs))
    r = np.fft.irfft((1.0 / eq) ** 2, FFT_LEN)
    reg = 1.0
    a = None
    while reg >= 0.7:
        a = levinson_durbin(r, taps, reg)
        if a is not None:
            break
        reg -= 2e-6
    if a is None:
        raise RuntimeError("LPC did not converge")
    mag = np.abs(np.fft.rfft(a, FFT_LEN))
    scale = 1.0 / mag[-1]
    return a * scale, float(np.max(mag * scale))


# --- per-channel design -----------------------------------------------------------------------------

CUTOFF_LARGE_BELOW = 40.0   # speakers with a detected cutoff below this are "Large" (no high pass)
MAX_EQ_DEFAULT = 24000.0
CIRRUS_DSP_MODELS = {
    "Marantz SR6013", "Marantz SR6014", "Denon AVR-X3400H", "Denon AVR-X3500H", "Denon AVR-X3600H", "Marantz SR5013",
    "Marantz SR5014", "Marantz NR1608", "Marantz NR1609", "Denon AVR-X2400H", "Denon AVR-X2500H", "Denon AVR-X2600H",
    "Denon AVR-S930H", "Denon AVR-S940H", "Denon AVR-S950H", "Denon AVR-X1400H", "Denon AVR-X1500H", "Denon AVR-X1600H",
    "Denon AVR-S730H", "Denon AVR-S740H", "Denon AVR-S750H", "Marantz NR1710", "Marantz SR5012", "*AVR-S720W",
    "*AVR-S920W", "*AVR-X1300W", "*AVR-X2300W", "*AVR-X3300W", "*NR1607", "*SR5011", "Denon AVR-S960H",
    "Denon AVR-X2700H", "Marantz SR5015", "Marantz NR1711", "Denon AVR-S760H", "Denon AVR-X1700H", "Denon AVR-S970H",
    "Denon AVR-X2800H", "Marantz CINEMA 70s", "Marantz CINEMA 60", "Denon AVR-S770H", "Denon AVR-X1800H",
    "Denon AVR-S980H", "Denon AVR-X2900H", "Marantz CINEMA 60 SERIES 2", "Marantz CINEMA 70s SERIES 2",
}
SAT_TAPS_XT = 512


def uses_acm1h_mic(model):
    """Receivers whose measurements need the ACM1H mic correction applied by the app."""
    return model in CIRRUS_DSP_MODELS or model.startswith("Denon AVC-X2850H")


class Channel:
    """Measurement-derived state of one channel: averaged response, reference position, cutoff."""

    def __init__(self, irs, mic_spec, is_sub=False):
        self.is_sub = is_sub
        self.weighting = level_weighting(fft_freqs(), SUB_WEIGHT_BAND if is_sub else SAT_WEIGHT_BAND)
        acc = np.zeros(BINS)
        self.position_levels = []
        for ir in irs:
            mag = response_magnitude(ir, mic_spec)
            lvl = weighted_level(mag, self.weighting)
            self.position_levels.append(lvl)
            acc += mag / lvl
        self.avg = acc / weighted_level(acc, self.weighting)
        # reference position: first measured position, without mic correction, scaled by its level
        x = np.zeros(FFT_LEN)
        ir0 = np.asarray(irs[0], dtype=np.float64)[:FFT_LEN]
        x[:len(ir0)] = ir0
        self.pos_resp = np.abs(np.fft.rfft(x)) / self.position_levels[0]
        self.cutoff, self.cutoff_order = detect_cutoff(self.avg, is_sub)


def design_eq(ch, points, base_curve, midrange, cutoff_option, max_eq, fs):
    """EQ magnitude (0..24 kHz grid) for one channel and curve, as used for filter design at rate fs."""
    freqs = fft_freqs()
    effective_cutoff = max(ch.cutoff, cutoff_option)
    target = target_curve(freqs, points, base_curve, midrange, cutoff_option, effective_cutoff)
    nyq = fs * 0.5
    f_top = max_eq if max_eq <= nyq else nyq
    hf_start = hf_limit_start(f_top)
    taper = interp_bezier([hf_start, f_top], [1.0, 0.0], freqs)
    hf_idx = frequency_to_index(hf_start)
    eq0 = np.exp(smooth_log(np.log(target / ch.avg), SMOOTH_SUB if ch.is_sub else SMOOTH_SAT))
    hi, lo = eq_limits(freqs, effective_cutoff, cutoff_option)
    return level_eq(eq0, ch.pos_resp, hi, lo, taper, hf_idx, ch.weighting)


def design_filter(ch, points, base_curve, midrange, cutoff_option, max_eq, fs, taps=SAT_TAPS_XT):
    eq = design_eq(ch, points, base_curve, midrange, cutoff_option, max_eq, fs)
    coefs, peak = lpc_filter(eq, taps, fs)
    max_boost = 10.0 ** (MAX_BOOST_DB / 20.0)
    if peak > max_boost:
        coefs = coefs * (max_boost / peak)
    return coefs


# --- .ady front end --------------------------------------------------------------------------------

CHANNEL_TYPE_NAMES = {
    0: "FrontLeft", 1: "Center", 2: "FrontRight", 3: "FrontWideRight", 4: "SurrRight", 5: "SurrRightA", 6: "SurrRightB",
    7: "SurrRightC", 8: "SBackRight", 9: "SBackCenter", 10: "SBackLeft", 11: "SurrLeftC", 12: "SurrLeftB",
    13: "SurrLeftA", 14: "SurrLeft", 15: "FrontWideLeft", 16: "FrontHeightRight", 18: "TopFrontRight",
    19: "TopMiddleRight", 30: "TopMiddleLeft", 31: "TopFrontLeft", 33: "FrontHeightLeft",
}
RATES = (32000, 44100, 48000)
FLOAT_MIN = 1.4e-45


def _f32_array(values):
    return np.asarray([float(v) for v in values], dtype=np.float32)


def _irs(channel):
    irs = []
    for key in sorted(channel["responseData"], key=int):
        v = _f32_array(channel["responseData"][key])
        v[np.abs(v) > 10.0] = np.float32(FLOAT_MIN)
        irs.append(v.astype(np.float64))
    return irs


def _with_edge_points(points):
    """The app adds (20 Hz, 0 dB) / (20 kHz, 0 dB) when no point lies near the band edges."""
    pts = [tuple(p) for p in points]
    if pts and not any(f <= np.sqrt(2.0) * 20.0 for f, _ in pts):
        pts.insert(0, (20.0, 0.0))
    if pts and not any(f >= 20000.0 / np.sqrt(2.0) for f, _ in pts):
        pts.append((20000.0, 0.0))
    return pts


def _partner(name):
    if "Left" in name:
        return name.replace("Left", "Right")
    if "Right" in name:
        return name.replace("Right", "Left")
    return None


def compute_filters(ady, rates=RATES):
    """Reference + flat filters for every channel of an .ady (path or dict); same structure as
    meq_filter.compute_filters (satellite channels of MultEQ XT receivers only).

    Per commandId: channelType, cutoffHz, speakerSize, trimDb, appLevelDb, distanceMs, appDistanceM,
    reference/flat: {fs: {"ok", "raw"}}, referenceDisplay/flatDisplay: {"display61"} (48 kHz filter).
    Distances come from the receiver's own measurement in the .ady (channelReport.distance).
    """
    from meq_protocol import app_distance, app_level, parse_points
    if not isinstance(ady, dict):
        import json
        with open(ady) as f:
            ady = json.load(f)
    if ady.get("enMultEQType", 1) != 1:
        raise ValueError("only MultEQ XT is supported")
    model = ady.get("targetModelName") or ""
    acm1h = uses_acm1h_mic(model)
    mic = mic_spectrum(table("mic_acm1h")) if acm1h else np.ones(BINS, dtype=complex)
    ref_curve = {0: 0, 1: 1, 2: 2}[ady.get("enTargetCurveType", 1)]
    srcs = ady["detectedChannels"]
    names = [CHANNEL_TYPE_NAMES.get(c["enChannelType"], str(c["enChannelType"])) for c in srcs]
    if any(n.startswith("SW") or not n[0].isalpha() for n in names):
        raise ValueError("subwoofer / unknown channel types are not supported")
    irs = [_irs(c) for c in srcs]
    chans = [Channel(ir, mic) for ir in irs]
    out = {"libVersion": "meq_design"}
    for i, (src, ch) in enumerate(zip(srcs, chans)):
        cutoff = ch.cutoff
        partner = _partner(names[i])
        if partner in names:
            cutoff = max(cutoff, chans[names.index(partner)].cutoff)
        cutoff_option = 0.0 if cutoff < CUTOFF_LARGE_BELOW else cutoff
        rolloff = src.get("frequencyRangeRolloff", 20000.0)
        max_eq = MAX_EQ_DEFAULT if rolloff == 20000.0 else rolloff
        midrange = bool(src.get("midrangeCompensation", True))
        points = _with_edge_points(parse_points(src.get("customTargetCurvePoints")))
        res = {"index": i, "channelType": names[i], "cutoffHz": cutoff,
               "speakerSize": "Large" if cutoff < CUTOFF_LARGE_BELOW else "Small"}
        for key, base, meq in (("reference", ref_curve, max_eq), ("flat", 0, MAX_EQ_DEFAULT)):
            res[key] = {fs: {"ok": True, "raw": design_filter(ch, points, base, midrange, cutoff_option, meq, fs).tolist()}
                        for fs in rates}
            shown = res[key][FS]["raw"] if FS in res[key] else design_filter(ch, points, base, midrange, cutoff_option, meq, FS)
            res[key + "Display"] = {"display61": filter_display(shown)}
        res["trimDb"] = float(channel_trim(irs[i][0], mic, ady.get("adcLineup", 0.0), acm1h=acm1h))
        res["appLevelDb"] = app_level(res["trimDb"], src.get("trimAdjustment", 0.0))
        distance_m = float((src.get("channelReport") or {}).get("distance", 0.0))
        res["distanceMs"] = distance_m * 1000.0 / 344.3
        res["appDistanceM"] = app_distance(res["distanceMs"], src.get("delayAdjustment", 0.0))
        out[src.get("commandId", names[i])] = res
    return out


DISPLAY_RANGE = (20.0, 20000.0)


def filter_display(coefs, points=61, fs=FS):
    """Filter magnitude in dB at `points` log-spaced frequencies from 20 Hz to 20 kHz (SET_DISFIL data)."""
    f = np.geomspace(DISPLAY_RANGE[0], DISPLAY_RANGE[1], points)
    n = np.arange(len(coefs))
    h = np.exp(-2j * np.pi * np.outer(f / fs, n)) @ np.asarray(coefs, dtype=np.float64)
    return (20.0 * np.log10(np.maximum(np.abs(h), 1e-6))).tolist()


# --- channel level (trim) -------------------------------------------------------------------------

TRIM_REFERENCE_DB = 105.0
SPL_OFFSET_DB = 93.9794            # 20*log10(50000)
PREAMP_GAIN_DB = 34.6              # as passed by the app
MIC_SENSITIVITY = 0.01122          # ACM1H
SUB_TRIM_WEIGHT = 0.8912509        # -1 dB


def _db(x):
    return 20.0 * np.log10(x)


def channel_trim(ir, mic_spec, adc_lineup, is_sub=False, acm1h=True):
    """Level trim in dB for one measurement position (the app uses position 0)."""
    freqs = fft_freqs()
    if is_sub:
        w = SUB_TRIM_WEIGHT * level_weighting(freqs, SUB_WEIGHT_BAND)
    else:
        w = level_weighting(freqs, SAT_WEIGHT_BAND)
    lvl = rms(response_magnitude(ir, mic_spec) * w)
    sens = MIC_SENSITIVITY
    if not acm1h:
        # without the ACM1H response the app normalizes the sensitivity by the mic table at 1 kHz
        sens = MIC_SENSITIVITY / np.abs(mic_spectrum(table("mic_acm1h")))[frequency_to_index(1000.0)]
    offset = _db(adc_lineup) - PREAMP_GAIN_DB - _db(sens) + SPL_OFFSET_DB
    return TRIM_REFERENCE_DB - (_db(lvl) + offset)


def main():
    import argparse
    import json
    p = argparse.ArgumentParser(description="Design MultEQ XT filters for an .ady file (pure Python).")
    p.add_argument("ady")
    p.add_argument("-o", "--out", help="write results (raw coefficients) as JSON")
    p.add_argument("--rates", default=",".join(map(str, RATES)))
    args = p.parse_args()
    res = compute_filters(args.ady, rates=[int(r) for r in args.rates.split(",")])
    for cid, ch in res.items():
        if cid == "libVersion":
            continue
        print(f"{cid:4s} {ch['channelType']:11s} {ch['speakerSize']:5s} cutoff {ch['cutoffHz']:6.1f} Hz  "
              f"level {ch['appLevelDb']:+5.1f} dB  dist {ch['appDistanceM']:.2f} m")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f)


if __name__ == "__main__":
    main()
