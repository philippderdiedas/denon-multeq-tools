#!/usr/bin/env python3
"""ady.py - read, inspect and edit Audyssey MultEQ Editor (.ady) files.

Stdlib only, runs on Linux and Windows (Python 3.8+).

An .ady file is JSON. Unknown fields are preserved on round-trip, so the
tool only touches what you explicitly change.

Examples:
    python ady.py info setup.ady
    python ady.py curve setup.ady
    python ady.py notch setup.ady -o fixed.ady --freq 87 --gain -10 --q 8
    python ady.py set-point setup.ady -o out.ady --channel FL --freq 120 --db -2
    python ady.py remove-point setup.ady -o out.ady --freq 120
    python ady.py clear-curve setup.ady -o out.ady --channel SW1
    python ady.py set setup.ady -o out.ady --channel C --level -1.5 --distance 2.4
    python ady.py export-curve setup.ady --channel FL > fl.txt
    python ady.py import-curve setup.ady fl.txt -o out.ady --channel FL
    python ady.py export-ir setup.ady --outdir ir/
"""

import argparse
import json
import math
import os
import struct
import sys

SAMPLE_RATE = 48000  # Audyssey impulse responses are 48 kHz
CURVE_KEY = "customTargetCurvePoints"


# ---------------------------------------------------------------- file I/O

def load(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        raw = f.read()
    data = json.loads(raw)
    # Remember the original layout so we write back in the same style.
    data_style = "pretty" if "\n" in raw.strip() else "compact"
    return data, data_style


def save(data, path, style):
    if style == "pretty":
        text = json.dumps(data, indent=2, ensure_ascii=False)
    else:
        text = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"wrote {path}", file=sys.stderr)


def channels(data):
    chans = data.get("detectedChannels")
    if not isinstance(chans, list):
        sys.exit("error: no 'detectedChannels' list found - is this an .ady file?")
    return chans


def channel_name(ch, idx):
    for key in ("commandId", "channelName", "enChannelType"):
        if key in ch:
            return str(ch[key])
    return f"#{idx}"


def select_channels(data, spec):
    """spec: None/'all' -> every channel, else comma list of names or indices."""
    chans = channels(data)
    if not spec or spec.lower() == "all":
        return list(enumerate(chans))
    wanted = [s.strip() for s in spec.split(",") if s.strip()]
    out = []
    for w in wanted:
        hit = [(i, c) for i, c in enumerate(chans)
               if channel_name(c, i).lower() == w.lower() or str(i) == w.lstrip("#")]
        if not hit:
            names = ", ".join(channel_name(c, i) for i, c in enumerate(chans))
            sys.exit(f"error: channel '{w}' not found (have: {names})")
        out.extend(hit)
    return out


# ---------------------------------------------------------- target curves

def parse_curve(points):
    """Curve points are stored as strings like "{87.1, -10}"."""
    curve = []
    for p in points or []:
        if isinstance(p, str):
            f, db = (float(x) for x in p.replace("{", "").replace("}", "").split(","))
        else:
            f, db = float(p[0]), float(p[1])
        curve.append((f, db))
    return sorted(curve)


def num(x):
    """Format like the app does: 163.1, 0, -20, 0.1."""
    return f"{round(x, 1):g}"


def format_curve(curve):
    return [f"{{{num(f)}, {num(db)}}}" for f, db in sorted(curve)]


def get_curve(ch):
    return parse_curve(ch.get(CURVE_KEY))


def set_curve(ch, curve):
    ch[CURVE_KEY] = format_curve(curve)


def interp_curve(curve, f):
    """Linear interpolation over log-frequency; flat 0 dB if curve is empty."""
    if not curve:
        return 0.0
    if f <= curve[0][0]:
        return curve[0][1]
    if f >= curve[-1][0]:
        return curve[-1][1]
    for (f0, d0), (f1, d1) in zip(curve, curve[1:]):
        if f0 <= f <= f1:
            t = (math.log(f) - math.log(f0)) / (math.log(f1) - math.log(f0))
            return d0 + t * (d1 - d0)
    return 0.0


def peaking_db(f, fc, gain, q, fs=SAMPLE_RATE):
    """Magnitude (dB) of an RBJ peaking EQ biquad at frequency f."""
    a = 10 ** (gain / 40)
    w0 = 2 * math.pi * fc / fs
    alpha = math.sin(w0) / (2 * q)
    b = (1 + alpha * a, -2 * math.cos(w0), 1 - alpha * a)
    den = (1 + alpha / a, -2 * math.cos(w0), 1 - alpha / a)
    w = 2 * math.pi * f / fs
    z1, z2 = complex(math.cos(-w), math.sin(-w)), complex(math.cos(-2 * w), math.sin(-2 * w))
    h = (b[0] + b[1] * z1 + b[2] * z2) / (den[0] + den[1] * z1 + den[2] * z2)
    return 20 * math.log10(abs(h))


def notch_points(fc, gain, q, density):
    """Sample the peaking filter shape around fc as target curve points."""
    bw_oct = 2 / math.log(2) * math.asinh(1 / (2 * q))
    span = 2 * bw_oct                   # cover +-2 bandwidths
    pts = []
    for k in range(-density, density + 1):
        f = fc * 2 ** (span * k / density)
        pts.append((f, peaking_db(f, fc, gain, q)))
    return pts, fc * 2 ** -span, fc * 2 ** span


# ---------------------------------------------------------------- commands

INFO_KEYS = ("title", "targetModelName", "interfaceVersion", "enMultEQType",
             "enAmpAssignType", "dynamicEq", "dynamicVolume", "lfcSupport", "lfc",
             "systemDelay", "adcLineup", "enTargetCurveType", "auro", "upgradeInfo")

CHANNEL_KEYS = ("trimAdjustment", "delayAdjustment", "customLevel", "customDistance",
                "customCrossover", "customSpeakerType", "midrangeCompensation",
                "frequencyRangeRolloff", "isSkipMeasurement")


def cmd_info(args):
    data, _ = load(args.file)
    for k in INFO_KEYS:
        if k in data:
            v = data[k]
            if isinstance(v, str) and len(v) > 60:
                v = v[:57] + "..."
            print(f"{k:22} {v}")
    print()
    for i, ch in enumerate(channels(data)):
        name = channel_name(ch, i)
        report = ch.get("channelReport") or {}
        fields = "  ".join(f"{k.replace('custom', '')}={ch[k]}" for k in CHANNEL_KEYS
                           if ch.get(k) is not None)
        if "distance" in report:
            fields = f"measured={report['distance']}m  " + fields
        if report.get("isReversePolarity"):
            fields += "  REVERSE-POLARITY"
        npts = len(ch.get(CURVE_KEY) or [])
        nmeas = len(ch.get("responseData") or {})
        print(f"[{i}] {name:6} {fields}  curvePoints={npts}  measurements={nmeas}")


def cmd_curve(args):
    data, _ = load(args.file)
    for i, ch in select_channels(data, args.channel):
        print(f"[{i}] {channel_name(ch, i)}")
        curve = get_curve(ch)
        if not curve:
            print("    (no custom target curve)")
        for f, db in curve:
            print(f"    {f:9.1f} Hz  {db:+6.1f} dB")


def cmd_notch(args):
    data, style = load(args.file)
    new_pts, lo, hi = notch_points(args.freq, args.gain, args.q, args.density)
    for i, ch in select_channels(data, args.channel):
        curve = get_curve(ch)
        merged = [(f, db) for f, db in curve if not lo <= f <= hi]
        merged += [(f, round(interp_curve(curve, f) + d, 1)) for f, d in new_pts]
        set_curve(ch, merged)
        print(f"{channel_name(ch, i)}: notch {args.freq} Hz {args.gain:+} dB Q{args.q} "
              f"({lo:.1f}-{hi:.1f} Hz, {len(new_pts)} points)", file=sys.stderr)
    save(data, args.output, style)


def cmd_set_point(args):
    data, style = load(args.file)
    for _, ch in select_channels(data, args.channel):
        curve = [(f, db) for f, db in get_curve(ch) if abs(f - args.freq) > 0.05]
        curve.append((args.freq, args.db))
        set_curve(ch, curve)
    save(data, args.output, style)


def cmd_remove_point(args):
    data, style = load(args.file)
    for i, ch in select_channels(data, args.channel):
        curve = get_curve(ch)
        kept = [(f, db) for f, db in curve if abs(f - args.freq) > args.tolerance]
        print(f"{channel_name(ch, i)}: removed {len(curve) - len(kept)} point(s)", file=sys.stderr)
        set_curve(ch, kept)
    save(data, args.output, style)


def cmd_clear_curve(args):
    data, style = load(args.file)
    for _, ch in select_channels(data, args.channel):
        ch[CURVE_KEY] = []
    save(data, args.output, style)


def cmd_set(args):
    data, style = load(args.file)
    for i, ch in select_channels(data, args.channel):
        for opt, key in (("level", "customLevel"), ("distance", "customDistance"),
                         ("crossover", "customCrossover"), ("speaker_type", "customSpeakerType")):
            val = getattr(args, opt)
            if val is not None:
                print(f"{channel_name(ch, i)}: {key} {ch.get(key)} -> {val}", file=sys.stderr)
                ch[key] = val
        if args.midrange_compensation is not None:
            ch["midrangeCompensation"] = args.midrange_compensation == "on"
    save(data, args.output, style)


def cmd_export_curve(args):
    data, _ = load(args.file)
    _, ch = select_channels(data, args.channel)[0]
    for f, db in get_curve(ch):
        print(f"{f:.1f} {db:.1f}")


def cmd_import_curve(args):
    """Text format: one 'freq dB' pair per line (REW / Equalizer APO style)."""
    curve = []
    with open(args.curve, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.split("#")[0].split("*")[0].strip()
            parts = line.replace(",", " ").replace(";", " ").split()
            if len(parts) >= 2:
                try:
                    curve.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue
    if not curve:
        sys.exit(f"error: no 'freq dB' lines found in {args.curve}")
    data, style = load(args.file)
    for _, ch in select_channels(data, args.channel):
        set_curve(ch, curve)
    save(data, args.output, style)


def write_wav_float(path, samples, fs=SAMPLE_RATE):
    payload = struct.pack(f"<{len(samples)}f", *samples)
    fmt = struct.pack("<HHIIHH", 3, 1, fs, fs * 4, 4, 32)  # IEEE float mono
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 4 + 8 + len(fmt) + 8 + len(payload)) + b"WAVE")
        f.write(b"fmt " + struct.pack("<I", len(fmt)) + fmt)
        f.write(b"data" + struct.pack("<I", len(payload)) + payload)


def cmd_export_ir(args):
    """Dump measured impulse responses as WAV - importable in REW."""
    data, _ = load(args.file)
    os.makedirs(args.outdir, exist_ok=True)
    n = 0
    for i, ch in select_channels(data, args.channel):
        resp = ch.get("responseData") or {}
        items = resp.items() if isinstance(resp, dict) else enumerate(resp)
        for pos, samples in items:
            path = os.path.join(args.outdir, f"{channel_name(ch, i)}_pos{pos}.wav")
            write_wav_float(path, [float(s) for s in samples])
            n += 1
    print(f"wrote {n} impulse response(s) to {args.outdir}", file=sys.stderr)


# --------------------------------------------------------------------- cli

def main():
    p = argparse.ArgumentParser(description="Inspect and edit Audyssey .ady files.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_, out=True, chan=True):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("file", help=".ady input file")
        if out:
            sp.add_argument("-o", "--output", required=True, help="output .ady file")
        if chan:
            sp.add_argument("-c", "--channel", default="all",
                            help="channel name(s) or index, comma separated (default: all)")
        sp.set_defaults(fn=fn)
        return sp

    add("info", cmd_info, "show settings and channels", out=False, chan=False)
    add("curve", cmd_curve, "show custom target curve points", out=False)

    sp = add("notch", cmd_notch, "cut a narrow band into the target curve")
    sp.add_argument("--freq", type=float, required=True, help="center frequency (Hz)")
    sp.add_argument("--gain", type=float, default=-10.0, help="gain in dB (default -10)")
    sp.add_argument("--q", type=float, default=8.0, help="Q factor (default 8)")
    sp.add_argument("--density", type=int, default=4,
                    help="points per side of the notch (default 4)")

    sp = add("set-point", cmd_set_point, "add or replace a target curve point")
    sp.add_argument("--freq", type=float, required=True)
    sp.add_argument("--db", type=float, required=True)

    sp = add("remove-point", cmd_remove_point, "remove target curve point(s) near a frequency")
    sp.add_argument("--freq", type=float, required=True)
    sp.add_argument("--tolerance", type=float, default=0.5, help="Hz (default 0.5)")

    add("clear-curve", cmd_clear_curve, "delete all custom target curve points")

    sp = add("set", cmd_set, "change channel level / distance / crossover")
    sp.add_argument("--level", type=float, help="trim in dB")
    sp.add_argument("--distance", type=float, help="distance in m")
    sp.add_argument("--crossover", type=int, help="crossover in Hz")
    sp.add_argument("--speaker-type", help="e.g. Small / Large")
    sp.add_argument("--midrange-compensation", choices=("on", "off"))

    add("export-curve", cmd_export_curve, "print one channel's curve as 'freq dB' text",
        out=False)
    sp = add("import-curve", cmd_import_curve, "replace curve from a 'freq dB' text file")
    sp.add_argument("curve", help="text file with 'freq dB' lines")

    sp = add("export-ir", cmd_export_ir, "export measured impulse responses as WAV", out=False)
    sp.add_argument("--outdir", default="ir", help="output directory (default ./ir)")

    args = p.parse_args()
    if getattr(args, "output", None) and os.path.abspath(args.output) == os.path.abspath(args.file):
        if os.environ.get("ADY_ALLOW_OVERWRITE") != "1":
            sys.exit("error: refusing to overwrite the input file; pick another -o "
                     "(or set ADY_ALLOW_OVERWRITE=1)")
    args.fn(args)


if __name__ == "__main__":
    main()
