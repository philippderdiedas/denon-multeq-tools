#!/usr/bin/env python3
"""avr_transfer.py - upload an Audyssey .ady calibration to a Denon/Marantz receiver over the LAN.

The receiver's CURRENT distances, levels, crossovers and speaker sizes are read via telnet and sent
back unchanged, so a transfer only replaces the EQ filters. Frame formats live in meq_protocol.py;
the filter design comes from a pluggable backend (see --backend).

    python avr_transfer.py plan  my.ady --host 192.168.1.50          # dry run (read-only queries)
    python avr_transfer.py send  my.ady --host 192.168.1.50 --yes    # really write

Tested on: Denon AVR-X1400H, MultEQ XT, 5.0 (FL/C/FR/SLA/SRA), no subwoofer. Everything else is
refused unless --experimental is given. Use at your own risk; see README.
"""

import argparse
import importlib
import json
import socket
import struct
import sys
import time

import meq_protocol as proto

AVR_PORT = 1256
TELNET_PORT = 23
ACK_TIMEOUT = 5.0
ACK_HARD_LIMIT = 60.0            # INPROGRESS replies must not extend the wait forever

# commandId (calibration port) -> telnet channel name and speaker group (SSCFR/SSSPC)
CHANNELS = {
    "FL": ("FL", "FRO"), "FR": ("FR", "FRO"), "C": ("C", "CEN"),
    "SLA": ("SL", "SUA"), "SRA": ("SR", "SUA"),
    "SBL": ("SBL", "SBK"), "SBR": ("SBR", "SBK"), "SB": ("SB", "SBK"),
    "FHL": ("FHL", "FRH"), "FHR": ("FHR", "FRH"),
}
TESTED_CHANNELS = {"FL", "FR", "C", "SLA", "SRA"}
SIZE = {"LAR": "L", "SMA": "S"}


# ------------------------------------------------------------------ network

class Calibration:
    """Connection to the receiver's calibration port."""

    def __init__(self, host):
        self.sock = socket.create_connection((host, AVR_PORT), timeout=ACK_TIMEOUT)
        self.buf = b""

    def close(self):
        self.sock.close()

    def _next_frame(self, deadline):
        while True:
            if len(self.buf) >= 3:
                total = struct.unpack(">H", self.buf[1:3])[0]
                if len(self.buf) >= total:
                    frame, self.buf = self.buf[:total], self.buf[total:]
                    return proto.parse_frame(frame)
            remaining = deadline - time.time()
            if remaining <= 0:
                return None
            self.sock.settimeout(remaining)
            try:
                chunk = self.sock.recv(65536)
            except socket.timeout:
                return None
            if not chunk:
                raise RuntimeError("connection closed by receiver")
            self.buf += chunk

    def query(self, command):
        """Send a GET_* command and return its JSON reply."""
        self.sock.sendall(proto.build_frame(command))
        reply = self._next_frame(time.time() + ACK_TIMEOUT)
        if not reply or reply["command"] != command:
            raise RuntimeError(f"{command}: no reply")
        return json.loads(reply["params"])

    def send(self, frame, label):
        """Send one frame and wait for ACK; NACK or garbage raises."""
        self.sock.sendall(frame)
        hard = time.time() + ACK_HARD_LIMIT
        deadline = min(time.time() + ACK_TIMEOUT, hard)
        while True:
            reply = self._next_frame(deadline)
            if reply is None:
                raise RuntimeError(f"{label}: no ACK")
            if not reply["checksum_ok"]:
                raise RuntimeError(f"{label}: corrupt reply")
            status, detail = proto.ack_status(reply["params"])
            if status == "ACK":
                return
            if status == "INPROGRESS":
                deadline = min(time.time() + ACK_TIMEOUT, hard)
                continue
            raise RuntimeError(f"{label}: receiver answered {status}" + (f" ({detail})" if detail else ""))


def telnet(host, commands):
    """Send telnet commands, return {command: [reply lines]}."""
    s = socket.create_connection((host, TELNET_PORT), timeout=3)
    out = {}
    try:
        for c in commands:
            s.sendall(c.encode() + b"\r")
            time.sleep(0.6)
            s.settimeout(0.4)
            buf = b""
            try:
                while True:
                    d = s.recv(4096)
                    if not d:
                        break
                    buf += d
            except socket.timeout:
                pass
            out[c] = [l for l in buf.decode(errors="replace").split("\r") if l]
    finally:
        s.close()
    return out


def read_state(host):
    """Read-only snapshot of everything the transfer depends on."""
    cal = Calibration(host)
    try:
        info = cal.query("GET_AVRINF")
        time.sleep(0.2)
        status = cal.query("GET_AVRSTS")
    finally:
        cal.close()
    tel = telnet(host, ["SSSDE ?", "SSLEV ?", "SSCFR ?", "SSSPC ?",
                        "PSDYNEQ ?", "PSDYNVOL ?", "PSREFLEV ?", "PSMULTEQ: ?"])
    return {"info": info, "status": status, "telnet": tel}


def prepare_receiver(host):
    """Before a transfer: main zone on, close the on-screen menu."""
    if "ZMOFF" in "".join(telnet(host, ["ZM?"])["ZM?"]):
        telnet(host, ["ZMON"])
        time.sleep(10)
    telnet(host, ["MNMEN ON"])
    time.sleep(0.2)
    telnet(host, ["MNMEN OFF"])
    time.sleep(0.5)


# ------------------------------------------------------------------- checks

def speaker_settings(state, channels):
    """Current distance (m), level (dB), size and crossover per commandId, from telnet."""
    lines = [l for v in state["telnet"].values() for l in v]

    def pick(prefix):
        return {l[len(prefix):].split(" ")[0]: l.split(" ")[-1] for l in lines
                if l.startswith(prefix) and " " in l}

    dist, lev, cfr, spc = pick("SSSDE"), pick("SSLEV"), pick("SSCFR"), pick("SSSPC")
    out = {}
    for cid in channels:
        tname, group = CHANNELS[cid]
        raw = lev[tname]
        size = SIZE[spc[group]]
        xo = cfr[group]
        if size == "S" and not xo.isdigit():
            raise ValueError(f"{cid}: unexpected crossover {xo!r} for a small speaker")
        out[cid] = {"distance_m": int(dist[tname]) / 100,
                    # levels are 50 = 0 dB, three digits mean half-dB steps (495 = -0.5 dB)
                    "level_db": int(raw) / (10 if len(raw) == 3 else 1) - 50,
                    "size": size,
                    # Large speakers may report non-numeric values (e.g. TRR); "F" is sent for them
                    "crossover": "F" if size == "L" else str(int(xo))}
    return out


def check_supported(info, status, ady, experimental):
    problems = []
    if info.get("EQType") != "MultEQXT":
        problems.append(f"MultEQ type {info.get('EQType')} (only MultEQXT is implemented)")
    if info.get("DType") != "FixedA":
        problems.append(f"coefficient data type {info.get('DType')} (only FixedA is tested)")
    order = [ch["commandId"] for ch in ady["detectedChannels"]]
    avr = [k for e in status["ChSetup"] for k, v in e.items() if v != "N"]
    if sorted(order) != sorted(avr):
        raise ValueError(f"channel mismatch: .ady {order} vs receiver {avr} - fix the speaker setup first")
    if ady.get("ampAssignInfo") != status.get("AssignBin"):
        raise ValueError("amp assignment in .ady differs from the receiver - recalibrate with the current setup")
    subs = [c for c in order if c.startswith("SW")]
    if subs:
        raise ValueError(f"subwoofers {subs} are not supported yet")
    unknown = [c for c in order if c not in CHANNELS]
    if unknown:
        raise ValueError(f"channels {unknown} are not supported yet")
    untested = [c for c in order if c not in TESTED_CHANNELS]
    if untested:
        problems.append(f"untested channels {untested}")
    if problems and not experimental:
        raise ValueError("refusing: " + "; ".join(problems) + " (use --experimental at your own risk)")
    return order


# --------------------------------------------------------------------- plan

def load_backend(name):
    try:
        return importlib.import_module(name)
    except ImportError as e:
        sys.exit(f"filter backend '{name}' not available: {e}")


def build_plan(ady_path, state, backend, experimental=False):
    """Every frame of the transfer in order: [(label, frame or None, wait_ms)]."""
    info, status = state["info"], state["status"]
    with open(ady_path, encoding="utf-8-sig") as f:
        ady = json.load(f)
    order = check_supported(info, status, ady, experimental)
    spk = speaker_settings(state, order)

    # Round-trip through JSON so sample-rate keys are strings in every backend.
    filters = json.loads(json.dumps(backend.compute_filters(ady)))
    setting = proto.app_setting_data(
        ady, filters,
        crossover={c: spk[c]["crossover"] for c in order},
        distance={c: spk[c]["distance_m"] for c in order},
        level={c: spk[c]["level_db"] for c in order})
    for cid, size in setting["SpConfig"]:
        if size != spk[cid]["size"]:
            raise ValueError(f"{cid}: Audyssey wants size {size}, receiver has {spk[cid]['size']}")

    waits = info.get("CoefWaitTime", {})
    plan = [("ENTER_AUDY", proto.build_frame("ENTER_AUDY"), 0)]
    for i, p in enumerate(proto.setdat_packets(setting)):
        plan.append((f"SET_SETDAT {i + 1}", proto.build_frame("SET_SETDAT", p), 0))
    for curve in ("reference", "flat"):
        for cid in order:
            disp = filters[cid][f"{curve}Display"]["display61"]
            plan.append((f"SET_DISFIL {curve} {cid}",
                         proto.build_frame("SET_DISFIL", proto.disfil_payload(disp, curve, cid)), 0))
    if waits.get("Init", 0):
        plan.append(("INIT_COEFS", proto.build_frame("INIT_COEFS"), 2000))
    for fs in proto.RATES:
        for curve in ("reference", "flat"):
            for cid in order:
                res = filters[cid][curve][str(fs)]
                if not res["ok"]:
                    raise RuntimeError(f"filter design failed for {cid} {curve} {fs}")
                payload = proto.encode_coefdt(res["raw"], channel=cid, fs=fs, curve=curve,
                                              target_curve_type=ady.get("enTargetCurveType", 1),
                                              data_type=info["DType"])
                last, chunks = proto.split_packets(payload)
                for n, chunk in enumerate(chunks):
                    plan.append((f"SET_COEFDT {curve} {cid} {fs} {n + 1}/{len(chunks)}",
                                 proto.build_frame("SET_COEFDT", chunk, n, last), 0))
    # Finalizing is only requested by receivers that report a final wait time.
    if waits.get("Final", 0):
        plan.append(("FINZ_COEFS", proto.build_frame("FINZ_COEFS"), 0))
    plan.append(("SET_SETDAT Fin", proto.build_frame("SET_SETDAT", '{"AudyFinFlg":"Fin"}'), 0))
    plan.append(("EXIT_AUDMD", proto.build_frame("EXIT_AUDMD"), 0))
    return plan, filters, setting


def verify_plan(plan, filters, data_type):
    """Re-split every SET_COEFDT stream and compare it with a fresh encoding."""
    streams = {}
    for label, frame, _ in plan:
        assert sum(frame[:-1]) & 0xFF == frame[-1], f"{label}: bad checksum"
        assert struct.unpack(">H", frame[1:3])[0] == len(frame), f"{label}: bad length"
        if label.startswith("SET_COEFDT"):
            _, curve, cid, fs, _ = label.split(" ")
            plen = struct.unpack(">H", frame[16:18])[0]
            streams.setdefault((curve, cid, fs), []).append(frame[18:18 + plen])
    for (curve, cid, fs), parts in streams.items():
        expected = proto.encode_coefdt(filters[cid][curve][fs]["raw"], channel=cid, fs=int(fs),
                                       curve=curve, data_type=data_type)
        assert b"".join(parts) == expected, f"payload mismatch {curve} {cid} {fs}"
    return len(streams)


# ---------------------------------------------------------------------- cli

def cmd_plan(args):
    state = json.load(open(args.state)) if args.state else read_state(args.host)
    if args.save_state:
        json.dump(state, open(args.save_state, "w"), indent=2)
    plan, filters, setting = build_plan(args.file, state, load_backend(args.backend), args.experimental)
    n = verify_plan(plan, filters, state["info"]["DType"])
    for key in ("SpConfig", "Distance", "ChLevel", "Crossover"):
        print(f"  {key:9} {setting[key]}")
    print("  " + "  ".join(f"{k}={setting[k]}" for k in ("AudyDynEq", "AudyDynVol", "AudyEqSet")))
    print(f"{len(plan)} frames, {sum(len(f) for _, f, _ in plan)} bytes, {n} filter streams verified")
    if args.dump:
        with open(args.dump, "w") as f:
            for label, frame, wait in plan:
                f.write(f"{label}\t{wait}\t{frame.hex()}\n")
        print(f"wrote {args.dump}")


def cmd_send(args):
    if not args.yes:
        sys.exit("refusing to write to the receiver without --yes (try 'plan' first)")
    state = read_state(args.host)
    plan, filters, _ = build_plan(args.file, state, load_backend(args.backend), args.experimental)
    verify_plan(plan, filters, state["info"]["DType"])
    prepare_receiver(args.host)
    cal = Calibration(args.host)
    entered = False
    try:
        for i, (label, frame, wait) in enumerate(plan):
            time.sleep(wait / 1000)
            print(f"[{i + 1}/{len(plan)}] {label}", flush=True)
            cal.send(frame, label)
            entered = True
            time.sleep(0.02)
        print("done")
    except (Exception, KeyboardInterrupt) as e:
        print(f"TRANSFER FAILED: {e!r}", file=sys.stderr)
        if entered:
            try:
                cal.send(proto.build_frame("EXIT_AUDMD"), "EXIT_AUDMD")
                print("left calibration mode - the old filters may be partly overwritten, "
                      "run the transfer again", file=sys.stderr)
            except Exception as e2:
                print(f"could not leave calibration mode ({e2}); run 'exit' or power-cycle the receiver",
                      file=sys.stderr)
        sys.exit(1)
    finally:
        cal.close()


def cmd_exit(args):
    """Leave calibration mode (e.g. after an interrupted transfer)."""
    cal = Calibration(args.host)
    try:
        cal.send(proto.build_frame("EXIT_AUDMD"), "EXIT_AUDMD")
        print("left calibration mode")
    finally:
        cal.close()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("plan", cmd_plan), ("send", cmd_send)):
        sp = sub.add_parser(name)
        sp.add_argument("file", help=".ady calibration file")
        sp.add_argument("--host", required=True, help="receiver IP address")
        sp.add_argument("--backend", default="meq_design", help="filter design module (default meq_design)")
        sp.add_argument("--experimental", action="store_true", help="allow untested receivers/channels")
        sp.set_defaults(fn=fn)
    sub.choices["plan"].add_argument("--state", help="use a saved receiver state instead of querying")
    sub.choices["plan"].add_argument("--save-state", help="save the queried receiver state")
    sub.choices["plan"].add_argument("--dump", help="write every frame as hex to this file")
    sub.choices["send"].add_argument("--yes", action="store_true", help="really write to the receiver")
    sp = sub.add_parser("exit", help="leave calibration mode")
    sp.add_argument("--host", required=True)
    sp.set_defaults(fn=cmd_exit)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
