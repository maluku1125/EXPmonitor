#!/usr/bin/env python3
"""
exp_monitor_debug.py -- OCR debug session recorder
====================================================
每幀儲存所有中間圖片 + 模擬主程式驗證層，幫助找出誤報原因。

Usage:
    python exp_monitor_debug.py
    python exp_monitor_debug.py --interval 0.5 --frames 200

Output: debug_YYYYMMDD_HHMMSS/
    session.csv
    NNNNN_raw.png         原始截圖條
    NNNNN_annotated.png   標注 EXP row 位置
    NNNNN_exprow.png      裁切的 EXP row
    NNNNN_HSV.png         HSV 遮罩 (5x)
    NNNNN_Otsu.png        Otsu 二值化 (5x)
    NNNNN_Bright.png      亮度遮罩 (5x)
"""

import os, sys, cv2, csv, time, json, argparse, importlib.util
import numpy as np
from datetime import datetime
from pathlib import Path

# ── 載入核心模組 ──────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("exp_core", _HERE / "exp_monitor.py")
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

find_window        = _mod.find_window
capture_strip      = _mod.capture_strip
find_exp_bar_rows  = _mod.find_exp_bar_rows
find_exp_text_cols = _mod.find_exp_text_cols
find_fill_boundary = _mod.find_fill_boundary
preprocess         = _mod.preprocess
run_ocr            = _mod.run_ocr
parse              = _mod.parse
_setup_tess        = _mod._setup_tess
_init_easy         = _mod._init_easy
set_dpi_awareness  = _mod.set_dpi_awareness
_has_exp_yellow    = _mod._has_exp_yellow

# ── CLI ───────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--interval", "-i", type=float, default=1.0)
ap.add_argument("--frames",   "-n", type=int,   default=0)
ap.add_argument("--threshold","-t", type=float, default=1.0,
                help="Layer-1 max_exp 容差 %（與主程式相同，預設 1.0）")
args = ap.parse_args()

# ── Session ───────────────────────────────────────────────────────────────────
ts_start   = datetime.now()
session_id = ts_start.strftime("%Y%m%d_%H%M%S")
out_dir    = _HERE / f"debug_{session_id}"
out_dir.mkdir(parents=True, exist_ok=True)

CSV_FIELDS = [
    "frame", "timestamp",
    "status",
    "cap_method", "has_yellow",
    "row_y0", "row_y1",
    "best_mask", "best_raw",
    "pct", "exp",
    "cur_max_exp", "max_exp_est", "max_exp_dev_pct",
    "val_result",
    "all_masks",
]

# ── OCR init ──────────────────────────────────────────────────────────────────
set_dpi_awareness()
use_tess = _setup_tess()
if not use_tess:
    _init_easy()

print(f"\n{'='*68}")
print(f"  EXP Monitor Debug  --  {session_id}")
print(f"  OCR: {'Tesseract' if use_tess else 'EasyOCR'}  |  interval={args.interval}s  |  threshold={args.threshold}%")
print(f"  Output: {out_dir}")
print(f"{'='*68}")
print(f"  {'#':>5}  {'time':12}  {'OCR':5}  {'Y':1}  {'row':5}  {'pct':10}  {'exp':14}  {'validate'}")
print(f"{'─'*68}")

# ── 模擬驗證狀態 ──────────────────────────────────────────────────────────────
sim_max_exp_est  = None
sim_prev_exp_int = None
sim_prev_pct     = None

def simulate_validation(pct_str, exp_str):
    global sim_max_exp_est, sim_prev_exp_int, sim_prev_pct

    result = "pass"
    cur_max = None
    dev_pct = ""
    max_est_str = ""

    try:
        pct_f   = float(pct_str)
        exp_int = int(exp_str.replace(",", "")) if exp_str else None

        if exp_int and pct_f > 1.0:
            cur_max = exp_int / (pct_f / 100.0)

            if sim_max_exp_est is not None:
                dev = abs(cur_max - sim_max_exp_est) / sim_max_exp_est
                dev_pct     = f"{dev*100:.4f}%"
                max_est_str = f"{sim_max_exp_est/1e12:.3f}T"
                if dev > args.threshold / 100.0:
                    result = f"L1 maxexp偏差{dev*100:.2f}%"

            if result == "pass" and sim_prev_exp_int and exp_int < sim_prev_exp_int:
                result = f"L2 EXP減少 {sim_prev_exp_int - exp_int:,}"

            if result == "pass":
                if sim_max_exp_est is None:
                    sim_max_exp_est = cur_max
                else:
                    sim_max_exp_est = sim_max_exp_est * 0.9 + cur_max * 0.1
                sim_prev_exp_int = exp_int
                sim_prev_pct     = pct_f

    except Exception as e:
        result = f"err:{e}"

    cur_max_str = f"{cur_max/1e12:.3f}T" if cur_max else ""
    return result, cur_max_str, max_est_str, dev_pct


def fmt_exp(s):
    if not s:
        return "—"
    try:
        v = int(s.replace(",", ""))
        return f"{v/1e9:.3f}B" if v >= 1e9 else f"{v:,}"
    except Exception:
        return s


# ── Main loop ─────────────────────────────────────────────────────────────────
frame    = 0
ok_count = 0

with open(out_dir / "session.csv", "w", newline="", encoding="utf-8-sig") as csvf:
    writer = csv.DictWriter(csvf, fieldnames=CSV_FIELDS)
    writer.writeheader()

    try:
        while True:
            frame += 1
            now_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            rec = {k: "" for k in CSV_FIELDS}
            rec["frame"]     = frame
            rec["timestamp"] = now_str

            # 1. Capture
            hwnd, reg = find_window()
            if hwnd is None:
                rec["status"] = "no_window"
                print(f"  {frame:>5}  {now_str}  NO WINDOW")
                writer.writerow(rec)
                csvf.flush()
                if args.frames and frame >= args.frames:
                    break
                time.sleep(args.interval)
                continue

            img, cap_lbl = capture_strip(hwnd, reg)
            rec["cap_method"] = cap_lbl or "unknown"

            if img is None:
                rec["status"] = "cap_fail"
                print(f"  {frame:>5}  {now_str}  CAP FAIL ({cap_lbl})")
                writer.writerow(rec)
                csvf.flush()
                if args.frames and frame >= args.frames:
                    break
                time.sleep(args.interval)
                continue

            # 2. Yellow check + save raw
            has_yel = _has_exp_yellow(img)
            rec["has_yellow"] = "1" if has_yel else "0"
            cv2.imwrite(str(out_dir / f"{frame:05d}_raw.png"), img)

            # 3. EXP row detection
            y0, y1 = find_exp_bar_rows(img)
            rec["row_y0"] = y0
            rec["row_y1"] = y1

            ann = img.copy()
            cv2.rectangle(ann, (0, y0), (img.shape[1]-1, max(y1-1, y0)), (0, 255, 0), 1)
            cv2.imwrite(str(out_dir / f"{frame:05d}_annotated.png"), ann)

            text_band = img[y0:y1, :] if y1 > y0 else img
            x0, x1   = find_exp_text_cols(text_band, img.shape[1])
            exp_row   = text_band[:, x0:x1]
            rec["row_y0"] = f"{y0}(x{x0})"
            rec["row_y1"] = f"{y1}(x{x1})"
            cv2.imwrite(str(out_dir / f"{frame:05d}_exprow.png"), exp_row)

            # 4. Preprocessing + OCR per mask
            mask_results = []
            for label, mask in preprocess(exp_row):
                cv2.imwrite(str(out_dir / f"{frame:05d}_{label}.png"), mask)
                raw_txt, _ = run_ocr(mask, use_tess)
                # parse() returns (exp_integer_str, pct_str)
                exp_v, pct_v = parse(raw_txt) if raw_txt else (None, None)
                mask_results.append({
                    "mask": label,
                    "raw":  raw_txt or "",
                    "pct":  pct_v  or "",
                    "exp":  exp_v  or "",
                })

            rec["all_masks"] = json.dumps(mask_results, ensure_ascii=False)

            # 5. Best result
            def score(r):
                return 2 if r["pct"] and r["exp"] else 1 if r["pct"] else 0

            best = max(mask_results, key=score)
            rec["best_mask"] = best["mask"]
            rec["best_raw"]  = best["raw"]
            rec["pct"]       = best["pct"]
            rec["exp"]       = best["exp"]

            if best["pct"]:
                ok_count += 1
                rec["status"] = "ok"
                ocr_tag = "OK   "
            else:
                rec["status"] = "fail"
                ocr_tag = "FAIL "

            # 6. Simulate Layer-1 + Layer-2 validation
            val_result, cur_max_str, max_est_str, dev_pct = simulate_validation(
                best["pct"], best["exp"])
            rec["cur_max_exp"]      = cur_max_str
            rec["max_exp_est"]      = max_est_str
            rec["max_exp_dev_pct"]  = dev_pct
            rec["val_result"]       = val_result

            yel_ch   = "Y" if has_yel else "N"
            row_str  = f"{y0}-{y1}"
            pct_str  = f"{best['pct']}%" if best["pct"] else f"({best['raw'][:8]!r})"
            exp_str  = fmt_exp(best["exp"])
            val_str  = "V:OK" if val_result == "pass" else f"V:NG {val_result[:22]}"
            print(f"  {frame:>5}  {now_str}  {ocr_tag}  {yel_ch}  {row_str:5}  {pct_str:10}  {exp_str:14}  {val_str}")

            writer.writerow(rec)
            csvf.flush()

            if args.frames and frame >= args.frames:
                break
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("Stopped.")

elapsed  = (datetime.now() - ts_start).total_seconds()
fail_ocr = frame - ok_count
rate_ocr = ok_count / frame * 100 if frame else 0
print(f"Total: {frame}  OK: {ok_count} ({rate_ocr:.1f}%)  Output: {out_dir}")
