# denon-multeq-tools

Tools for Denon/Marantz receivers with **Audyssey MultEQ XT**: inspect and edit `.ady`
calibration files, find room/furniture resonances with a test tone, and upload the
calibration to the receiver over the LAN without the mobile app.

Built to tame a desk panel that rattled between 160 and 300 Hz. The fix was a custom target
curve on the two channels that excited it, plus a bass shelf to keep the low end full.

> **Status: experimental.** Tested on exactly one receiver: Denon AVR-X1400H (MultEQ XT,
> FixedA, 5.0, no subwoofer). Anything else is refused unless you pass `--experimental`.
> Not affiliated with or endorsed by Audyssey, Denon, Marantz or Masimo. Use at your own risk.

## What's inside

| File | What it does |
|---|---|
| `ady-editor.html` | Browser app (no install). Open an `.ady`, see measured response and target curve, drag curve points, add notches, play a sine/sweep on selected channels to find resonances, save. |
| `ady.py` | CLI for the same edits: `info`, `curve`, `notch`, `set-point`, `import-curve`, `export-ir` (impulse responses as WAV for REW), ... |
| `avr_transfer.py` | Uploads an `.ady` to the receiver (port 1256). Dry run by default. |
| `meq_protocol.py` | Frame and payload formats of the calibration protocol. |
| `meq_design.py` | MultEQ XT filter design (512-tap minimum-phase FIR per channel). |
| `meq_data/` | Target curves and microphone correction used by the filter design (see *Data*). |

Python 3.8+ and the standard library are enough for the editor, `ady.py` and the protocol code.
The filter design (and therefore `avr_transfer.py`) also needs numpy:

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt     # Windows: .venv\Scripts\pip ...
```

## Workflow

1. **Measure** with the Audyssey MultEQ Editor app and export the `.ady` file. These tools
   cannot measure (yet).
2. **Find the problem.** Open the file in `ady-editor.html`, tick one channel and sweep the
   test tone. Windows must output 5.1/7.1 to the receiver, otherwise only FL/FR are reachable.
   Don't use *Direct*/*Pure Direct*, they bypass Audyssey.
3. **Edit the target curve**: notches, shelves, or copy a curve to other channels. Save.
4. **Dry run**, which only reads from the receiver:
   ```
   python avr_transfer.py plan my_edit.ady --host 192.168.1.50
   ```
   It shows exactly what will be sent: speaker sizes, distances, levels, crossovers and the
   Dynamic EQ/Volume flags, all taken from the receiver's *current* state, plus all filters.
5. **Upload**:
   ```
   python avr_transfer.py send my_edit.ady --host 192.168.1.50 --yes
   ```
   It takes 1–2 minutes and the receiver is muted meanwhile. If the transfer is interrupted,
   run `python avr_transfer.py exit --host ...`, then send again.

## Things you should know

- **There is no backup.** Receivers cannot hand back their current filters. The protocol has no
  such command; we probed for one. To go back, upload your original `.ady` or rerun
  Audyssey from the receiver menu with the supplied microphone.
- **MultEQ XT filters are coarse in the bass.** 512 taps can't follow narrow notches below
  ~300 Hz. Expect a broad dip a bit wider than what you drew. The editor shows your target,
  `avr_transfer.py plan` shows what the filter will really do.
- A transfer re-sends distances/levels/crossovers as the receiver reports them. For "Large"
  speakers the crossover is sent as full range, like the app does. The receiver may afterwards
  report odd values such as `TRR` for that setting.
- Not supported: subwoofers, MultEQ XT32, MultEQ (basic), float-coefficient receivers, more
  than 5 channels. Contributions with test reports are welcome.

## Data

`meq_data/` holds the Audyssey target curves (HFRO1/HFRO2, SMPTE, mid-range compensation, Dolby
elevation) and the ACM1H microphone correction. They are not covered by this project's license.

`meq_design.py` was checked against the app's filters on several real and 18 synthetic calibrations:
the frequency response differs by at most ~0.007 dB, and cutoffs, speaker sizes and levels are
identical. Distances are taken from the receiver's own measurement in the `.ady` file, which can
differ from the app by 1–2 cm.

## Credits

Protocol behaviour was established by interoperability analysis of the official app and by
observing a real receiver. OCA's A1 Evo project showed that a LAN transfer is possible at all.
No code from it is used here.

## License

WTFPL v2, see `LICENSE`. No warranty, see the top of that file. Audyssey, MultEQ, Denon and Marantz are trademarks of their respective owners.
