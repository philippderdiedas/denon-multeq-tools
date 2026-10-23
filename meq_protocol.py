#!/usr/bin/env python3
"""Denon/Marantz Audyssey calibration protocol: data formats used by the MultEQ Editor app (pure Python).

Covers the binary frame on the calibration port (1256), SET_COEFDT coefficient payloads, SET_SETDAT and
SET_DISFIL parameters, and the app's distance/level rounding. Behaviour mirrors the app for MultEQ / MultEQ XT
receivers; see the per-function notes for app quirks that are reproduced or optionally corrected.
"""
import json
import math
import re
import struct
from decimal import Decimal, ROUND_HALF_UP

RATES = (32000, 44100, 48000)
SAMPLE_RATE_CODES = {32000: 0, 44100: 1, 48000: 2}
PACKET_SIZE = 512
SETDAT_MAX_BYTES = 510

# Channel type ordinals as stored in .ady "enChannelType", by the commandId the receiver uses
COMMAND_ID_TO_CHANNEL_TYPE = {
    "FL": 0, "C": 1, "FR": 2, "SR": 4, "SRA": 5, "SBR": 8, "SB": 9, "SBL": 10, "SLA": 13, "SL": 14,
    "FHR": 16, "TFR": 18, "TMR": 19, "TML": 30, "TFL": 31, "FHL": 33,
}
COMMAND_ID_TO_NAME = {"FL": "FrontLeft", "C": "Center", "FR": "FrontRight", "SR": "SurrRight", "SRA": "SurrRightA",
                      "SBR": "SBackRight", "SB": "SBackCenter", "SBL": "SBackLeft", "SLA": "SurrLeftA",
                      "SL": "SurrLeft", "FHR": "FrontHeightRight", "FHL": "FrontHeightLeft"}

# SET_COEFDT header byte 2 (channel) for MultEQ / MultEQ XT receivers
HEADER_CHANNEL_CODES = {
    "FrontLeft": 0x00, "Center": 0x01, "FrontRight": 0x02, "FrontWideRight": 0x15,
    "SurrRight": 0x03, "SurrRightA": 0x03, "SurrRightB": 0x05, "SBackRight": 0x07,
    "SBackCenter": 0x08, "SBackLeft": 0x08, "SurrLeftB": 0x0A, "SurrLeftA": 0x0C, "SurrLeft": 0x0C,
    "FrontWideLeft": 0x1C, "FrontHeightLeft": 0x10, "FrontHeightCenter": 0x12, "FrontHeightRight": 0x14,
    "TopFrontRight": 0x04, "TopMiddleRight": 0x05, "TopBackRight": 0x06, "SurrHeightRight": 0x16,
    "RearHeightRight": 0x17, "Overhead": 0x1D, "RearHeightLeft": 0x1A, "SurrHeightLeft": 0x1B,
    "TopBackLeft": 0x09, "TopMiddleLeft": 0x0A, "TopFrontLeft": 0x0B, "FrontDolbyLeft": 0x1A,
    "FrontDolbyRight": 0x17, "SurrDolbyRight": 0x18, "SBDolbyRight": 0x18, "SurrDolbyLeft": 0x19,
    "SBDolbyLeft": 0x19, "SWLFE": 0x0D, "SWMix1": 0x0D, "SWMix2": 0x0E, "SWMix3": 0x21, "SWMix4": 0x22,
}

AMP_ASSIGN_NAMES = ("Normal", "Zone2", "Zone3", "Zone2_3", "BiAmp", "ZoneMono", "2chBiAmp", "2ch", "FrontHi", "FrontWi",
                    "FrontB", "9ch", "11ch", "PreAmp", "BiAmpZ2", "Free", "Atmos", "Auro3D", "TFront", "TMiddle",
                    "FDolby", "SDolby")
SETDAT_KEYS = ("AmpAssign", "AssignBin", "SpConfig", "Distance", "ChLevel", "Crossover", "AudyFinFlg", "AudyDynEq",
               "AudyEqRef", "AudyDynVol", "AudyDynSet", "AudyMultEq", "AudyEqSet", "AudyLfc", "AudyLfcLev", "SWSetup")


def f32(x):
    """Round a Python float to IEEE float32."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def java_round(x):
    """Math.round semantics: half rounds up."""
    return math.floor(x + 0.5)


def parse_points(points):
    """.ady customTargetCurvePoints ("{f, g}" strings or pairs) -> [[f, g], ...]."""
    out = []
    for p in points or []:
        if isinstance(p, str):
            out.append([float(x) for x in re.findall(r"-?[\d.]+(?:[eE]-?\d+)?", p)])
        else:
            out.append([float(p[0]), float(p[1])])
    return out


# --- frame on the calibration port ---------------------------------------------------------------

def build_frame(command, params=b"", seq=0, last=0):
    """'T' | total length (u16 BE, incl. checksum) | seq | last seq | 10-char command | 0x00 |
    parameter length (u16 BE) | parameters | checksum (sum of all preceding bytes, low 8 bits)."""
    if isinstance(params, str):
        params = params.encode()
    name = command.encode("ascii")
    total = 1 + 2 + 1 + 1 + len(name) + 1 + 2 + len(params) + 1
    body = b"T" + struct.pack(">HBB", total, seq, last) + name + b"\x00" + struct.pack(">H", len(params)) + params
    return body + bytes([sum(body) & 0xFF])


def parse_frame(data):
    """Parse a receiver reply ('R' frame). Returns dict(command, seq, last, cipher, params, length, checksum_ok)."""
    if not data or data[0] != ord("R"):
        return None
    total, seq, last = struct.unpack(">HBB", data[1:5])
    plen, = struct.unpack(">H", data[16:18])
    params = data[18:18 + plen]
    return {"command": data[5:15].decode("ascii", "replace"), "seq": seq, "last": last, "cipher": data[15],
            "params": params, "length": total,
            "checksum_ok": len(data) >= total and (sum(data[:total - 1]) & 0xFF) == data[total - 1]}


def ack_status(params):
    """Interpret a reply's JSON parameters: ("ACK" | "NACK" | "INPROGRESS" | "FORMAT", nack_status or None).
    NackStatus values seen in the app: "Mic_Off", "HP_On", "CanNotCom"."""
    try:
        obj = json.loads(params.decode("ascii") if isinstance(params, bytes) else params)
    except (ValueError, UnicodeDecodeError):
        return "FORMAT", None
    comm = obj.get("Comm") if isinstance(obj, dict) else None
    if comm in ("ACK", "INPROGRESS"):
        return comm, None
    if comm == "NACK":
        return "NACK", obj.get("NackStatus")
    return "FORMAT", None


# --- SET_COEFDT ----------------------------------------------------------------------------------

def app_scale(coefs):
    """The app's -9 dB transfer scaling: x * 10^-0.45, rounded half-up to 6 decimals, as float32."""
    factor = Decimal(math.pow(10.0, f32(-0.45)))
    out = []
    for x in coefs:
        if math.isnan(x) or math.isinf(x):
            out.append(0.0)
        else:
            out.append(f32(float((Decimal(f32(x)) * factor).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))))
    return out


def to_fixed_q31(x):
    """Q1.31 as the app encodes it: truncated magnitude, negatives as one's complement with sign bit."""
    mag = abs(x)
    i = 0x7FFFFFFF if mag >= 1.0 else int(math.floor(mag * 2147483648.0))
    if x < 0.0:
        i = (~i & 0xFFFFFFFF) | 0x80000000
    return i


def header(channel, fs, curve="reference", target_curve_type=1):
    """4-byte SET_COEFDT header: curve (0 = reference/HFRO, 1 = flat), rate code, channel code, 0."""
    name = COMMAND_ID_TO_NAME.get(channel, channel)
    is_hfro = curve == "reference" and target_curve_type in (1, 2)
    return bytes([0 if is_hfro else 1, SAMPLE_RATE_CODES[fs], HEADER_CHANNEL_CODES[name], 0])


def encode_coefdt(coefs, channel="FL", fs=48000, curve="reference", target_curve_type=1, data_type="FixedA", scale=True):
    """Full SET_COEFDT payload: header + little-endian coefficients.

    coefs: designed filter coefficients (scale=True applies the app's -9 dB scaling) or already-scaled ones.
    data_type: GET_AVRINF DType; "Float" -> float32, anything else -> Q1.31.
    """
    values = app_scale(coefs) if scale else [f32(c) for c in coefs]
    if data_type == "Float":
        body = b"".join(struct.pack("<f", v) for v in values)
    else:
        body = b"".join(struct.pack("<I", to_fixed_q31(v)) for v in values)
    return header(channel, fs, curve, target_curve_type) + body


def split_packets(payload, replicate_app_quirk=False):
    """Split a payload into 512-byte chunks. Returns (last_sequence_number, [chunk0, ...]).

    The app sends the last byte of every chunk as 0 (it copies one byte too few), which corrupts the MSB of
    coefficients at chunk boundaries. replicate_app_quirk=True reproduces that for byte-exact comparisons.
    """
    n = len(payload)
    last = n // PACKET_SIZE if n % PACKET_SIZE else n // PACKET_SIZE - 1
    chunks = []
    for seq in range(last + 1):
        size = (n % PACKET_SIZE) if (seq == last and n % PACKET_SIZE) else PACKET_SIZE
        chunk = bytearray(payload[seq * PACKET_SIZE: seq * PACKET_SIZE + size])
        if replicate_app_quirk:
            chunk[-1] = 0
        chunks.append(bytes(chunk))
    return last, chunks


# --- SET_DISFIL ----------------------------------------------------------------------------------

def disfil_payload(display61, curve, channel):
    """SET_DISFIL parameters from a channel's 61-point filter display (dB).

    FilData: the 61 points truncated to int. DispData: 9 band averages over points 7-13, 13-19, ... 49-55
    (7 points each, neighbouring bands share their edge point) and 56-60.
    """
    small = [java_round(sum(display61[a:b + 1]) / 7.0) for a, b in
             ((7, 13), (13, 19), (19, 25), (25, 31), (31, 37), (37, 43), (43, 49), (49, 55))]
    small.append(java_round(sum(display61[56:61]) / 5.0))
    eq_type = "Flat" if curve == "flat" else "Audy"
    large = ",".join(str(int(x)) for x in display61)
    return f'{{"EqType":"{eq_type}","ChData":"{channel}","FilData":[{large}],"DispData":[{",".join(map(str, small))}]}}'


# --- SET_SETDAT ----------------------------------------------------------------------------------

def _setdat_fragment(key, value):
    def pairs(items, fmt):
        return "[" + ",".join(f'{{"{k}":{fmt(v)}}}' for k, v in items) + "]"
    if key in ("AmpAssign", "AssignBin", "AudyFinFlg", "AudyDynSet", "AudyEqSet"):
        return f'"{key}":"{value}"'
    if key == "SpConfig":
        return f'"{key}":' + pairs(value, lambda v: f'"{v}"')
    if key == "Distance":  # meters -> cm
        return f'"{key}":' + pairs(value, lambda v: str(java_round(f32(v) * 100.0)))
    if key == "ChLevel":  # dB -> dB*10
        return f'"{key}":' + pairs(value, lambda v: str(java_round(f32(v) * 10.0)))
    if key == "Crossover":  # "F"/"U" quoted, codes like "80", "10" (=100 Hz), "25" (=250 Hz) as bare numbers
        return f'"{key}":' + pairs(value, lambda v: f'"{v}"' if v in ("F", "U") else str(v))
    if key in ("AudyDynEq", "AudyDynVol", "AudyMultEq", "AudyLfc"):
        return f'"{key}":' + ("true" if value else "false")
    if key == "SWSetup":
        return f'"{key}":{value}'
    return f'"{key}":{int(value)}'


def setdat_packets(setting, four_sw_model=False):
    """SET_SETDAT parameter strings, split like the app does.

    setting: dict with SETDAT_KEYS (SpConfig/Distance/ChLevel/Crossover as [(commandId, value), ...]).
    Fields are appended in SETDAT_KEYS order; a packet is closed before a field would make it >= 510 bytes.
    For models without four-subwoofer support SWSetup is skipped and the app's last packet ends in ",}";
    that is reproduced as-is because it is what receivers have been accepting.
    """
    packets, current, i = [], "{", 0
    while i < len(SETDAT_KEYS):
        key = SETDAT_KEYS[i]
        candidate = current if (key == "SWSetup" and not four_sw_model) else current + _setdat_fragment(key, setting[key])
        if len(candidate.encode()) >= SETDAT_MAX_BYTES:
            packets.append(current[:-1] + "}")
            current = "{"
            continue
        if i == len(SETDAT_KEYS) - 1:
            packets.append(candidate + "}")
        else:
            current = candidate + ","
        i += 1
    return packets


def crossover_code(speaker_size, cutoff_hz):
    """Default crossover for receivers up to model year 2019: Large -> "F", Small -> from the speaker cutoff."""
    if speaker_size == "Large":
        return "F"
    for hi, code in ((30, "F"), (50, "40"), (70, "60"), (85, "80"), (95, "90"), (105, "10"),
                     (115, "11"), (135, "12"), (175, "15"), (225, "20")):
        if 0 < cutoff_hz <= hi:
            return code
    return "25" if cutoff_hz > 225 else "U"


def app_distance(distance_ms, delay_adjust=0.0):
    """Speaker distance in meters: ms * 0.3443 - delayAdjust, clamped to 0..18, rounded to cm."""
    f = f32(distance_ms / 1000.0 * 344.3) - float(delay_adjust or 0.0)
    f = min(f, 18.0) if f >= 0.0 else 0.0
    return round(f * 100.0) * 0.01


def app_level(trim_db, trim_adjust=0.0):
    """Channel level in dB: trim + trimAdjust, rounded to 0.5 dB, clamped to +-12."""
    f = math.floor((trim_db + float(trim_adjust or 0.0)) * 2.0 + 0.5) * 0.5
    return max(-12.0, min(12.0, f + 0.0))


def app_setting_data(ady, filters, crossover=None, distance=None, level=None):
    """The SET_SETDAT values the app sends for an .ady and a design result (compute_filters dict structure).

    crossover/distance/level: optional {commandId: value} overrides (e.g. the receiver's current state).
    """
    chans = [c["commandId"] for c in ady["detectedChannels"]]
    size = {"Large": "L", "Small": "S"}
    xo = {cid: crossover_code(filters[cid]["speakerSize"], filters[cid]["cutoffHz"]) for cid in chans}
    xo.update(crossover or {})
    dist = {cid: filters[cid]["appDistanceM"] for cid in chans}
    dist.update(distance or {})
    lev = {cid: filters[cid]["appLevelDb"] for cid in chans}
    lev.update(level or {})
    return {
        "AmpAssign": AMP_ASSIGN_NAMES[ady.get("enAmpAssignType", 0)], "AssignBin": ady.get("ampAssignInfo", ""),
        "SpConfig": [(cid, size.get(filters[cid]["speakerSize"], "S")) for cid in chans],
        "Distance": [(cid, dist[cid]) for cid in chans], "ChLevel": [(cid, lev[cid]) for cid in chans],
        "Crossover": [(cid, xo[cid]) for cid in chans],
        "AudyFinFlg": "NotFin", "AudyDynEq": bool(ady.get("dynamicEq", True)), "AudyEqRef": 0,
        "AudyDynVol": bool(ady.get("dynamicVolume", False)), "AudyDynSet": "M", "AudyMultEq": True,
        "AudyEqSet": "Audy", "AudyLfc": bool(ady.get("lfc", False)), "AudyLfcLev": 4, "SWSetup": None,
    }
