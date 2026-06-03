"""
MapleStory EXP Monitor — UI
============================
執行：python exp_monitor_ui.py
"""

import ctypes
import os
import queue
import re
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, font as tkfont

# ── 從 exp_monitor.py 匯入核心邏輯 ──────────────────────────────────────────
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "exp_core", os.path.join(_HERE, "exp_monitor.py"))
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

set_dpi_awareness = _mod.set_dpi_awareness
find_window       = _mod.find_window
capture_strip     = _mod.capture_strip
find_exp_bar_rows = _mod.find_exp_bar_rows
preprocess        = _mod.preprocess
run_ocr           = _mod.run_ocr
parse             = _mod.parse
_setup_tess       = _mod._setup_tess
_init_easy        = _mod._init_easy


# ══════════════════════════════════════════════════════════════════════════
# 顏色主題
# ══════════════════════════════════════════════════════════════════════════
BG     = "#1a1a2e"
BG2    = "#16213e"
ACCENT = "#0f3460"
GREEN  = "#4ade80"
YELLOW = "#fbbf24"
RED    = "#f87171"
CYAN   = "#67e8f9"
GRAY   = "#6b7280"
WHITE  = "#f1f5f9"


# ══════════════════════════════════════════════════════════════════════════
# EXP 位數自動學習器
# ══════════════════════════════════════════════════════════════════════════
LOCK_REQUIRED = 5    # 連續幾筆相同才鎖定

class DigitGuard:
    """
    自動學習 EXP 數值的位數，連續 LOCK_REQUIRED 筆相同才鎖定。
    鎖定後若新讀取位數不符 → 視為誤報。
    若連續 LOCK_REQUIRED 筆出現新位數 → 重新鎖定（適應升級後的變化）。
    """
    def __init__(self):
        self._history: deque[int] = deque(maxlen=LOCK_REQUIRED)
        self._locked: int | None  = None

    def reset(self):
        self._history.clear()
        self._locked = None

    @property
    def locked(self) -> int | None:
        return self._locked

    @property
    def progress(self) -> int:
        """目前朝向鎖定目標累積的連續筆數。"""
        if not self._history:
            return 0
        target = self._history[-1]
        count  = 0
        for d in reversed(self._history):
            if d == target:
                count += 1
            else:
                break
        return count

    def check(self, exp_str: str | None) -> tuple[bool, str]:
        """
        傳入新讀取的 exp_str（可為 None）。
        回傳 (valid, reason)。
        - valid=True  → 位數合理，history 已更新
        - valid=False → 位數與鎖定值不符，history 不更新
        """
        if not exp_str:
            return True, ""   # 無 EXP 數字時不做位數驗證

        nd = len(exp_str.replace(",", ""))

        if self._locked is None:
            # 學習階段：先記錄
            self._history.append(nd)
            # 若最近 LOCK_REQUIRED 筆全相同 → 鎖定
            if len(self._history) == LOCK_REQUIRED and len(set(self._history)) == 1:
                self._locked = nd
            return True, ""
        else:
            if nd == self._locked:
                self._history.append(nd)
                return True, ""
            else:
                # 位數不符：暫時記錄，但此筆視為誤報
                # 若連續 LOCK_REQUIRED 筆偏差 → 重新鎖定（升級適應）
                self._history.append(nd)
                if len(set(self._history)) == 1:
                    # 已累積足夠多的新位數，重新鎖定
                    old = self._locked
                    self._locked = nd
                    return True, ""   # 這筆算有效（已重新學習完成）
                return False, f"位數 {nd} ≠ 鎖定值 {self._locked}"


# ══════════════════════════════════════════════════════════════════════════
# 驗證邏輯
# ══════════════════════════════════════════════════════════════════════════

def validate_pct(new_pct: float, prev_pct: float | None,
                 threshold: float) -> tuple[bool, str]:
    """
    只驗證百分比跳動，位數驗證由 DigitGuard 負責。
    - 下降超過 threshold → 誤報（升級歸零除外）
    - 暴增超過 threshold×3 → 誤報
    """
    if prev_pct is None:
        return True, ""
    diff        = new_pct - prev_pct
    is_levelup  = (new_pct < 5.0 and prev_pct > 90.0)
    if not is_levelup and diff < -threshold:
        return False, f"異常下降 {diff:+.3f}%"
    if diff > threshold * 3:
        return False, f"異常暴增 {diff:+.3f}%"
    return True, ""


# ══════════════════════════════════════════════════════════════════════════
# 背景監控執行緒
# ══════════════════════════════════════════════════════════════════════════

class MonitorThread(threading.Thread):
    def __init__(self, result_q: queue.Queue, cfg: dict):
        super().__init__(daemon=True)
        self.q    = result_q
        self.cfg  = cfg
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        use_tess = _setup_tess()
        if not use_tess:
            _init_easy()
        self.q.put(("status", f"OCR={'Tesseract' if use_tess else 'EasyOCR'} 就緒"))

        while not self._stop.is_set():
            try:
                self._tick(use_tess)
            except Exception as ex:
                self.q.put(("error", str(ex)))
            self._stop.wait(self.cfg.get("interval", 5))

    def _tick(self, use_tess):
        hwnd, reg = find_window()
        if hwnd is None:
            self.q.put(("no_window", None))
            return

        img, cap = capture_strip(hwnd, reg)
        if img is None:
            self.q.put(("cap_fail", cap))
            return

        y0, y1 = find_exp_bar_rows(img)
        row    = img[y0:y1, :]

        best_e, best_p, best_raw = None, None, ""
        for _, mask in preprocess(row):
            raw, _ = run_ocr(mask, use_tess)
            if not raw:
                continue
            e, p = parse(raw)
            sc = (1 if p else 0) + (1 if e else 0)
            bs = (1 if best_p else 0) + (1 if best_e else 0)
            if sc > bs:
                best_e, best_p, best_raw = e, p, raw
            if best_e and best_p:
                break

        ts = datetime.now().strftime("%H:%M:%S")
        if best_p:
            self.q.put(("reading", {
                "ts": ts, "pct": best_p,
                "exp": best_e, "raw": best_raw, "cap": cap,
            }))
        else:
            self.q.put(("ocr_fail", {"ts": ts, "raw": best_raw}))


# ══════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        set_dpi_awareness()
        self.title("楓之谷 EXP 監控")
        self.resizable(False, False)
        self.configure(bg=BG)

        self._thread: MonitorThread | None = None
        self._q        = queue.Queue()
        self._prev_pct: float | None = None
        self._prev_exp_int: int | None = None   # EXP 絕對值（用於遞增驗證）
        self._cfg      = {"interval": 5}
        self._guard    = DigitGuard()

        self._f_big   = tkfont.Font(family="Consolas",    size=28, weight="bold")
        self._f_med   = tkfont.Font(family="Consolas",    size=13)
        self._f_small = tkfont.Font(family="Consolas",    size=10)
        self._f_label = tkfont.Font(family="微軟正黑體",  size=10)
        self._f_title = tkfont.Font(family="微軟正黑體",  size=11, weight="bold")

        self._build_ui()
        self._poll()

    # ── 建構 UI ──────────────────────────────────────────────────────────

    def _build_ui(self):
        # ═══ 標題列 ══════════════════════════════════════════════════════
        hdr = tk.Frame(self, bg=ACCENT)
        hdr.pack(fill="x")
        tk.Label(hdr, text="🍁 楓之谷 EXP 監控",
                 bg=ACCENT, fg=WHITE, font=self._f_title
                 ).pack(side="left", padx=14, pady=8)
        self._status_dot = tk.Label(hdr, text="●", fg=GRAY,
                                     bg=ACCENT, font=("Segoe UI", 14))
        self._status_dot.pack(side="right", padx=6)
        self._status_lbl = tk.Label(hdr, text="待機",
                                     bg=ACCENT, fg=GRAY, font=self._f_label)
        self._status_lbl.pack(side="right")

        # ═══ 主顯示區 ════════════════════════════════════════════════════
        disp = tk.Frame(self, bg=BG2, pady=12)
        disp.pack(fill="x", padx=10, pady=(10, 4))

        tk.Label(disp, text="EXP 百分比",
                 bg=BG2, fg=GRAY, font=self._f_label).pack()
        self._pct_var = tk.StringVar(value="—")
        self._pct_lbl = tk.Label(disp, textvariable=self._pct_var,
                                  bg=BG2, fg=CYAN, font=self._f_big)
        self._pct_lbl.pack()

        tk.Label(disp, text="原始 EXP",
                 bg=BG2, fg=GRAY, font=self._f_label).pack(pady=(6,0))
        self._exp_var = tk.StringVar(value="—")
        tk.Label(disp, textvariable=self._exp_var,
                 bg=BG2, fg=WHITE, font=self._f_med).pack()

        # 位數學習狀態（自動，無需設定）
        self._digit_var = tk.StringVar(value="位數學習：學習中 0/5")
        tk.Label(disp, textvariable=self._digit_var,
                 bg=BG2, fg=GRAY, font=self._f_small).pack(pady=(4,0))

        # ═══ 設定區 ══════════════════════════════════════════════════════
        cfg_frame = tk.LabelFrame(self, text=" 設定 ",
                                   bg=BG, fg=GRAY, font=self._f_label,
                                   bd=1, relief="groove")
        cfg_frame.pack(fill="x", padx=10, pady=4)

        # 抓取間隔
        row1 = tk.Frame(cfg_frame, bg=BG)
        row1.pack(fill="x", padx=10, pady=5)
        tk.Label(row1, text="抓取間隔：",
                 bg=BG, fg=WHITE, font=self._f_label,
                 width=10, anchor="w").pack(side="left")
        self._interval_var = tk.IntVar(value=5)
        for s in [1, 2, 3, 5, 10, 30]:
            tk.Radiobutton(row1, text=f"{s}s",
                           variable=self._interval_var, value=s,
                           bg=BG, fg=WHITE, selectcolor=ACCENT,
                           activebackground=BG, activeforeground=CYAN,
                           font=self._f_small,
                           command=self._on_interval_change
                           ).pack(side="left", padx=3)
        tk.Label(row1, text="自訂:", bg=BG, fg=GRAY,
                 font=self._f_small).pack(side="left", padx=(8,2))
        self._custom_var = tk.StringVar()
        tk.Entry(row1, textvariable=self._custom_var,
                 width=4, bg=ACCENT, fg=WHITE,
                 insertbackground=WHITE,
                 font=self._f_small, relief="flat").pack(side="left")
        tk.Label(row1, text="s", bg=BG, fg=GRAY,
                 font=self._f_small).pack(side="left")
        tk.Button(row1, text="套用", bg=ACCENT, fg=WHITE, relief="flat",
                  font=self._f_small, cursor="hand2",
                  command=self._apply_custom_interval).pack(side="left", padx=4)

        # 誤報門檻（百分比跳動）
        row2 = tk.Frame(cfg_frame, bg=BG)
        row2.pack(fill="x", padx=10, pady=5)
        tk.Label(row2, text="誤報門檻：",
                 bg=BG, fg=WHITE, font=self._f_label,
                 width=10, anchor="w").pack(side="left")
        self._thresh_var = tk.DoubleVar(value=5.0)
        tk.Spinbox(row2, from_=0.1, to=50.0, increment=0.5,
                   textvariable=self._thresh_var,
                   width=6, bg=ACCENT, fg=WHITE,
                   buttonbackground=ACCENT,
                   insertbackground=WHITE,
                   font=self._f_small, relief="flat").pack(side="left")
        tk.Label(row2, text="% 差異視為誤報",
                 bg=BG, fg=GRAY, font=self._f_small).pack(side="left", padx=6)

        # ═══ 控制按鈕 ════════════════════════════════════════════════════
        btn_frame = tk.Frame(self, bg=BG)
        btn_frame.pack(pady=8)

        self._btn_start = tk.Button(
            btn_frame, text="▶  開始監控",
            bg="#166534", fg=WHITE, activebackground="#15803d",
            font=self._f_title, relief="flat", width=14, cursor="hand2",
            command=self._start)
        self._btn_start.pack(side="left", padx=6)

        self._btn_stop = tk.Button(
            btn_frame, text="■  停止",
            bg="#7f1d1d", fg=WHITE, activebackground="#991b1b",
            font=self._f_title, relief="flat", width=10, cursor="hand2",
            state="disabled", command=self._stop)
        self._btn_stop.pack(side="left", padx=6)

        tk.Button(btn_frame, text="清除紀錄",
                  bg=ACCENT, fg=GRAY, relief="flat",
                  font=self._f_small, cursor="hand2",
                  command=self._clear_log).pack(side="left", padx=6)

        # ═══ 紀錄面板 ════════════════════════════════════════════════════
        log_frame = tk.Frame(self, bg=BG)
        log_frame.pack(fill="both", expand=True, padx=10, pady=(0,10))

        tk.Label(log_frame, text="監控紀錄",
                 bg=BG, fg=GRAY, font=self._f_label).pack(anchor="w")

        self._log = tk.Text(
            log_frame, bg="#0f172a", fg=WHITE,
            font=self._f_small, height=12, width=72,
            state="disabled", relief="flat", wrap="none")
        self._log.pack(side="left", fill="both", expand=True)

        sb = ttk.Scrollbar(log_frame, command=self._log.yview)
        sb.pack(side="right", fill="y")
        self._log.configure(yscrollcommand=sb.set)

        self._log.tag_config("ok",   foreground=GREEN)
        self._log.tag_config("warn", foreground=YELLOW)
        self._log.tag_config("err",  foreground=RED)
        self._log.tag_config("info", foreground=GRAY)
        self._log.tag_config("ts",   foreground="#475569")
        self._log.tag_config("pct",  foreground=CYAN)
        self._log.tag_config("exp",  foreground=WHITE)
        self._log.tag_config("lock", foreground="#a78bfa")

    # ── 控制 ─────────────────────────────────────────────────────────────

    def _start(self):
        if self._thread and self._thread.is_alive():
            return
        self._prev_pct     = None
        self._prev_exp_int = None
        self._guard.reset()
        self._digit_var.set(f"位數學習：學習中 0/{LOCK_REQUIRED}")
        self._cfg["interval"] = self._interval_var.get()
        self._thread = MonitorThread(self._q, self._cfg)
        self._thread.start()
        self._btn_start.config(state="disabled")
        self._btn_stop.config(state="normal")
        self._set_status("監控中", GREEN)
        self._log_info("監控已啟動")

    def _stop(self):
        if self._thread:
            self._thread.stop()
            self._thread = None
        self._btn_start.config(state="normal")
        self._btn_stop.config(state="disabled")
        self._set_status("已停止", GRAY)
        self._log_info("監控已停止")

    def _on_interval_change(self):
        self._cfg["interval"] = self._interval_var.get()
        self._log_info(f"間隔改為 {self._cfg['interval']}s")

    def _apply_custom_interval(self):
        try:
            v = int(self._custom_var.get())
            if v < 1:
                raise ValueError
            self._cfg["interval"] = v
            self._interval_var.set(0)
            self._log_info(f"自訂間隔：{v}s")
        except ValueError:
            self._log_msg("⚠ 請輸入正整數", "warn")

    def _clear_log(self):
        self._log.config(state="normal")
        self._log.delete("1.0", "end")
        self._log.config(state="disabled")

    # ── queue 輪詢 ───────────────────────────────────────────────────────

    def _poll(self):
        try:
            while True:
                kind, data = self._q.get_nowait()
                self._handle(kind, data)
        except queue.Empty:
            pass
        self.after(200, self._poll)

    def _handle(self, kind, data):
        if kind == "reading":
            self._on_reading(data)
        elif kind == "no_window":
            self._set_status("找不到視窗", YELLOW)
            self._log_msg("找不到 MapleStory 視窗", "warn")
        elif kind == "cap_fail":
            self._set_status("截圖失敗", RED)
            self._log_msg(f"截圖失敗：{data}", "err")
        elif kind == "ocr_fail":
            self._set_status("OCR 失敗", YELLOW)
            self._log_msg(f"[{data['ts']}] OCR 無結果  raw={data['raw']!r}", "warn")
        elif kind == "status":
            self._log_info(str(data))
        elif kind == "error":
            self._log_msg(f"錯誤：{data}", "err")

    def _on_reading(self, d):
        ts  = d["ts"]
        pct = d["pct"]
        exp = d["exp"]
        cap = d["cap"]

        try:
            pct_f = float(pct)
        except Exception:
            pct_f = 0.0

        # ── 1. 百分比跳動驗證 ────────────────────────────────────────────
        pct_ok, pct_reason = validate_pct(
            pct_f, self._prev_pct, self._thresh_var.get())

        # ── 2. EXP 絕對值遞增驗證（經驗只加不減）────────────────────────
        exp_ok, exp_reason = True, ""
        exp_int = None
        if exp:
            try:
                exp_int = int(exp.replace(",", ""))
            except Exception:
                pass
        if exp_int is not None and self._prev_exp_int is not None:
            is_levelup = (pct_f < 5.0 and self._prev_pct is not None
                          and self._prev_pct > 90.0)
            if not is_levelup and exp_int < self._prev_exp_int:
                exp_ok     = False
                exp_reason = f"EXP 減少 {self._prev_exp_int - exp_int:,}"

        # ── 3. EXP 位數驗證（DigitGuard 自動學習）──────────────────────
        was_locked_before = self._guard.locked
        digit_ok, digit_reason = self._guard.check(exp)

        # 更新位數狀態顯示
        self._update_digit_display(was_locked_before)

        # ── 4. 合併判斷 ──────────────────────────────────────────────────
        valid  = pct_ok and exp_ok and digit_ok
        reason = pct_reason or exp_reason or digit_reason

        if valid:
            self._pct_var.set(f"{pct}%")
            self._pct_lbl.config(fg=CYAN)
            self._exp_var.set(exp or "—")
            self._set_status("監控中", GREEN)

            diff_str = ""
            if self._prev_pct is not None:
                diff = pct_f - self._prev_pct
                diff_str = f"  ({diff:+.3f}%)"

            self._log.config(state="normal")
            self._log.insert("end", f"[{ts}]  ", "ts")
            self._log.insert("end", f"{pct}%", "pct")
            if exp:
                self._log.insert("end", f"  {exp}", "exp")
            self._log.insert("end", f"{diff_str}  [{cap}]\n", "ok")
            self._log.config(state="disabled")
            self._log.see("end")

            self._prev_pct     = pct_f
            if exp_int is not None:
                self._prev_exp_int = exp_int

        else:
            self._pct_lbl.config(fg=YELLOW)
            self._set_status(f"誤報：{reason}", YELLOW)

            self._log.config(state="normal")
            self._log.insert("end", f"[{ts}]  ", "ts")
            self._log.insert("end", f"⚠ 誤報  pct={pct}%", "warn")
            if exp:
                self._log.insert("end", f"  exp={exp}", "warn")
            self._log.insert("end", f"  原因：{reason}\n", "warn")
            self._log.config(state="disabled")
            self._log.see("end")

    def _update_digit_display(self, was_locked_before):
        locked = self._guard.locked
        prog   = self._guard.progress

        if locked is None:
            self._digit_var.set(f"位數學習：學習中 {prog}/{LOCK_REQUIRED}")
        else:
            self._digit_var.set(f"位數學習：已鎖定 {locked} 位")
            # 若剛鎖定，印一行紀錄
            if was_locked_before != locked:
                ts_now = datetime.now().strftime("%H:%M:%S")
                self._log.config(state="normal")
                self._log.insert("end",
                    f"[{ts_now}]  🔒 EXP 位數已鎖定：{locked} 位\n", "lock")
                self._log.config(state="disabled")
                self._log.see("end")

    # ── 工具 ─────────────────────────────────────────────────────────────

    def _set_status(self, text, color):
        self._status_lbl.config(text=text, fg=color)
        self._status_dot.config(fg=color)

    def _log_info(self, msg):
        self._log_msg(f"ℹ {msg}", "info")

    def _log_msg(self, msg, tag="info"):
        self._log.config(state="normal")
        ts = datetime.now().strftime("%H:%M:%S")
        self._log.insert("end", f"[{ts}]  {msg}\n", tag)
        self._log.config(state="disabled")
        self._log.see("end")


# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = App()
    app.mainloop()
