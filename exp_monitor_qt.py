"""
MapleStory EXP Monitor — Qt UI v2
==================================
pip install PyQt5 pyqtgraph numpy
python exp_monitor_qt.py
"""

import os, sys, time
from collections import deque
from datetime import datetime

import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QDoubleSpinBox, QTextEdit,
    QButtonGroup, QRadioButton, QLineEdit, QFrame,
    QCheckBox, QScrollArea, QProgressBar, QSizePolicy, QSpinBox,
)
from PyQt5.QtCore import Qt, QThread, QObject, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtGui import QColor, QTextCursor, QFont
import pyqtgraph as pg

# ── Core module ──────────────────────────────────────────────────────────────
import importlib.util
_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "exp_core", os.path.join(_HERE, "exp_monitor.py"))
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

find_window         = _mod.find_window
capture_strip       = _mod.capture_strip
find_exp_bar_rows   = _mod.find_exp_bar_rows
find_exp_text_cols  = _mod.find_exp_text_cols
find_fill_boundary  = _mod.find_fill_boundary
preprocess          = _mod.preprocess
run_ocr             = _mod.run_ocr
parse               = _mod.parse
_setup_tess         = _mod._setup_tess
_init_easy          = _mod._init_easy
set_dpi_awareness   = _mod.set_dpi_awareness

try:
    from exp_template_ocr import TemplateOCR
    _HAS_TEMPLATE = True
except Exception:
    _HAS_TEMPLATE = False

# ══════════════════════════════════════════════════════════════════════════════
# 色彩
# ══════════════════════════════════════════════════════════════════════════════
C = {
    "bg":     "#0d1117",
    "bg2":    "#161b22",
    "bg3":    "#21262d",
    "border": "#30363d",
    "accent": "#1f6feb",
    "cyan":   "#58a6ff",
    "green":  "#3fb950",
    "yellow": "#e3b341",
    "red":    "#f85149",
    "gray":   "#8b949e",
    "white":  "#e6edf3",
    "purple": "#bc8cff",
    "plot_bg":"#0d1117",
}

QSS = f"""
* {{
    font-family: "微軟正黑體", "Segoe UI", sans-serif;
}}
QMainWindow, QWidget {{
    background: {C['bg']};
    color: {C['white']};
}}
QScrollArea, QScrollArea > QWidget > QWidget {{
    background: {C['bg']};
    border: none;
}}
QPushButton {{
    border-radius: 6px;
    padding: 7px 16px;
    font-size: 13px;
    font-weight: 600;
    border: 1px solid {C['border']};
    background: {C['bg3']};
    color: {C['white']};
}}
QPushButton:hover {{
    background: {C['accent']};
    border-color: {C['accent']};
}}
QPushButton:pressed {{
    background: #1158c7;
    border-color: #1158c7;
}}
QPushButton:disabled {{
    color: {C['gray']};
    border-color: {C['border']};
    background: {C['bg2']};
}}
QDoubleSpinBox, QLineEdit {{
    background: {C['bg3']};
    border: 1px solid {C['border']};
    border-radius: 6px;
    padding: 5px 10px;
    color: {C['white']};
    font-family: Consolas;
    font-size: 13px;
}}
QDoubleSpinBox:focus, QLineEdit:focus {{
    border-color: {C['accent']};
}}
QTextEdit {{
    background: {C['bg2']};
    border: 1px solid {C['border']};
    border-radius: 6px;
    color: {C['white']};
    font-family: Consolas;
    font-size: 12px;
    padding: 6px;
}}
QRadioButton {{
    color: {C['white']};
    font-size: 13px;
    spacing: 6px;
}}
QRadioButton::indicator {{
    width: 14px; height: 14px;
    border-radius: 7px;
    border: 2px solid {C['gray']};
    background: {C['bg']};
}}
QRadioButton::indicator:checked {{
    background: {C['accent']};
    border-color: {C['accent']};
}}
QCheckBox {{
    color: {C['white']};
    font-size: 13px;
    spacing: 8px;
}}
QCheckBox::indicator {{
    width: 16px; height: 16px;
    border-radius: 4px;
    border: 2px solid {C['gray']};
    background: {C['bg']};
}}
QCheckBox::indicator:checked {{
    background: {C['accent']};
    border-color: {C['accent']};
}}
QScrollBar:vertical {{
    background: {C['bg2']};
    width: 8px;
    border-radius: 4px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {C['bg3']};
    border-radius: 4px;
    min-height: 24px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ height: 0; }}
"""

# ══════════════════════════════════════════════════════════════════════════════
# EXP 位數守衛
# ══════════════════════════════════════════════════════════════════════════════
LOCK_REQUIRED = 5

class DigitGuard:
    def __init__(self):
        self._history = deque(maxlen=LOCK_REQUIRED)
        self._locked: int | None = None

    def reset(self):
        self._history.clear()
        self._locked = None

    @property
    def locked(self): return self._locked

    @property
    def progress(self):
        if not self._history: return 0
        target = self._history[-1]
        count = 0
        for d in reversed(self._history):
            if d == target: count += 1
            else: break
        return count

    def check(self, exp_str: str | None) -> tuple[bool, str]:
        if not exp_str: return True, ""
        nd = len(exp_str.replace(",", ""))
        if self._locked is None:
            self._history.append(nd)
            if len(self._history) == LOCK_REQUIRED and len(set(self._history)) == 1:
                self._locked = nd
            return True, ""
        if nd == self._locked:
            self._history.append(nd)
            return True, ""
        self._history.append(nd)
        if len(set(self._history)) == 1:
            self._locked = nd
            return True, ""
        return False, f"位數 {nd} ≠ {self._locked}"


# ══════════════════════════════════════════════════════════════════════════════
# 速率計算器
# ══════════════════════════════════════════════════════════════════════════════
class RateTracker:
    WIN_EPS = 60      # 1 分鐘 → EXP/s
    WIN_PPH = 3600    # 1 小時 → %/hr

    def __init__(self):
        self._eps_buf:     deque = deque()
        self._pph_buf:     deque = deque()
        self._all:         deque = deque()   # EXP% 歷史（無上限，由 chart_data 截斷）
        self._eps_samples: deque = deque()   # EXP/s 樣本（每秒一筆）

    def reset(self):
        self._eps_buf.clear()
        self._pph_buf.clear()
        self._all.clear()
        self._eps_samples.clear()

    def level_up_reset(self):
        """升等時清空速率緩衝，避免跨等計算失真。"""
        self._eps_buf.clear()
        self._pph_buf.clear()

    def add(self, exp_int: int, pct: float):
        now = time.time()
        self._all.append((now, exp_int, pct))
        self._eps_buf.append((now, exp_int, pct))
        self._pph_buf.append((now, exp_int, pct))
        eps_cut = now - self.WIN_EPS
        while self._eps_buf and self._eps_buf[0][0] < eps_cut:
            self._eps_buf.popleft()
        pph_cut = now - self.WIN_PPH
        while self._pph_buf and self._pph_buf[0][0] < pph_cut:
            self._pph_buf.popleft()

    def _rate_from(self, buf) -> tuple[float | None, float | None]:
        if len(buf) < 2: return None, None
        t0, e0, p0 = buf[0]; t1, e1, p1 = buf[-1]
        dt = t1 - t0
        if dt <= 0: return None, None
        return (e1 - e0) / dt, (p1 - p0) / dt * 3600

    @property
    def exp_per_sec(self) -> float | None:
        v, _ = self._rate_from(self._eps_buf); return v

    @property
    def pct_per_hour(self) -> float | None:
        _, v = self._rate_from(self._pph_buf); return v

    @property
    def sample_count(self) -> int:
        return len(self._all)

    def time_to_level(self, current_pct: float) -> str:
        pph = self.pct_per_hour
        if pph is None or pph <= 0: return "—"
        hours = (100.0 - current_pct) / pph
        if hours > 9999: return "> 9999h"
        h = int(hours); m = int((hours - h) * 60)
        return f"{h}h {m:02d}m"

    def chart_data(self, max_pts: int = 28800) -> tuple[list, list]:
        pts = list(self._all)[-max_pts:]
        if len(pts) < 2: return [], []
        t0 = pts[0][0]
        return ([(t - t0) / 60 for t, _, _ in pts],
                [p for _, _, p in pts])

    def add_eps_sample(self, eps: float):
        """每秒由 chart timer 呼叫，記錄當前 EXP/s 樣本。"""
        self._eps_samples.append((time.time(), eps))

    def eps_sample_data(self, max_pts: int = 28800) -> tuple[list, list]:
        """EXP/s 歷史（以分鐘為 X 軸）。"""
        pts = list(self._eps_samples)[-max_pts:]
        if len(pts) < 2: return [], []
        t0 = pts[0][0]
        return ([(t - t0) / 60 for t, _ in pts],
                [v for _, v in pts])

    def clear_chart_data(self):
        self._all.clear()
        self._eps_samples.clear()


# ══════════════════════════════════════════════════════════════════════════════
# 監控工作器
# ══════════════════════════════════════════════════════════════════════════════
class MonitorWorker(QObject):
    reading   = pyqtSignal(dict)
    no_window = pyqtSignal()
    cap_fail  = pyqtSignal(str)
    ocr_fail  = pyqtSignal(str)
    status    = pyqtSignal(str)
    error_sig = pyqtSignal(str)

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self._running = False
        self._use_tess = False
        self._ocr = None

    @pyqtSlot()
    def start_work(self):
        self._running  = True
        self._ocr = None
        if _HAS_TEMPLATE:
            o = TemplateOCR()
            if o.is_ready():
                self._ocr = o
        if self._ocr is not None:
            self._use_tess = False
            self.status.emit("OCR=Template(形狀比對) 就緒")
        else:
            self._use_tess = _setup_tess()
            if not self._use_tess:
                _init_easy()
            self.status.emit(f"OCR={'Tesseract' if self._use_tess else 'EasyOCR'} 就緒")
        self._loop()

    def stop(self): self._running = False

    def _loop(self):
        while self._running:
            try:
                self._tick()
            except Exception as ex:
                self.error_sig.emit(str(ex))
            interval = self.cfg.get("interval", 5)
            deadline = time.time() + interval
            while self._running and time.time() < deadline:
                time.sleep(0.05)

    def _tick(self):
        hwnd, reg = find_window()
        if hwnd is None:
            self.no_window.emit(); return
        img, cap = capture_strip(hwnd, reg)
        if img is None:
            self.cap_fail.emit(cap); return
        y0, y1    = find_exp_bar_rows(img)
        text_band = img[y0:y1, :]
        x0, x1   = find_exp_text_cols(text_band, img.shape[1])
        row       = text_band[:, x0:x1]
        best_e, best_p = None, None
        if self._ocr is not None:
            r = self._ocr.recognize_row(row)
            if r["exp"]:
                best_e = f"{int(r['exp']):,}"
            if r["pct"]:
                best_p = r["pct"]
        else:
            for _, mask in preprocess(row):
                raw, _ = run_ocr(mask, self._use_tess)
                if not raw: continue
                e, p = parse(raw)
                sc = (1 if p else 0) + (1 if e else 0)
                bs = (1 if best_p else 0) + (1 if best_e else 0)
                if sc > bs:
                    best_e, best_p = e, p
                if best_e and best_p:
                    break
        ts = datetime.now().strftime("%H:%M:%S")
        if best_p:
            self.reading.emit({"ts": ts, "pct": best_p, "exp": best_e, "cap": cap})
        else:
            self.ocr_fail.emit(ts)


# ══════════════════════════════════════════════════════════════════════════════
# UI 工具函式
# ══════════════════════════════════════════════════════════════════════════════
def _lbl(text, color=None, size=13, bold=False, mono=False):
    lb = QLabel(text)
    color = color or C["white"]
    w  = "600" if bold else "normal"
    ff = "Consolas" if mono else "'微軟正黑體','Segoe UI',sans-serif"
    lb.setStyleSheet(
        f"color:{color}; font-size:{size}px; font-weight:{w};"
        f" font-family:{ff}; background:transparent; border:none;")
    return lb


def _sep():
    line = QFrame()
    line.setFixedHeight(1)
    line.setStyleSheet(f"background:{C['border']}; border:none;")
    return line


def _card(pad_h=16, pad_v=14):
    f = QFrame()
    f.setStyleSheet(
        f"QFrame {{ background:{C['bg2']}; border:1px solid {C['border']};"
        f" border-radius:10px; }}")
    lay = QVBoxLayout(f)
    lay.setContentsMargins(pad_h, pad_v, pad_h, pad_v)
    lay.setSpacing(8)
    return f, lay


# ══════════════════════════════════════════════════════════════════════════════
# StatTile — 數值磚
# ══════════════════════════════════════════════════════════════════════════════
class StatTile(QFrame):
    def __init__(self, title: str, subtitle: str = "", value_size: int = 28):
        super().__init__()
        self._vsize = value_size
        self.setStyleSheet(
            f"QFrame {{ background:{C['bg3']}; border:1px solid {C['border']};"
            f" border-radius:8px; }}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 12)
        lay.setSpacing(2)

        title_row = QHBoxLayout()
        title_row.setSpacing(6)
        self._title_lb = _lbl(title, C["white"], 12, bold=True)
        title_row.addWidget(self._title_lb)
        if subtitle:
            title_row.addWidget(_lbl(subtitle, C["gray"], 10))
        title_row.addStretch()
        lay.addLayout(title_row)

        self._val_lb = _lbl("—", C["cyan"], value_size, bold=True, mono=True)
        lay.addWidget(self._val_lb)

    def set_value(self, text: str, color: str = None):
        color = color or C["cyan"]
        self._val_lb.setText(text)
        self._val_lb.setStyleSheet(
            f"color:{color}; font-size:{self._vsize}px; font-weight:600;"
            f" font-family:Consolas; background:transparent; border:none;")


# ══════════════════════════════════════════════════════════════════════════════
# SettingsPanel — 可折疊設定面板
# ══════════════════════════════════════════════════════════════════════════════
class SettingsPanel(QFrame):
    vis_changed = pyqtSignal()

    def __init__(self, cfg: dict, vis: dict, parent=None):
        super().__init__(parent)
        self._cfg = cfg
        self._vis = vis
        self.setStyleSheet(
            f"QFrame {{ background:{C['bg2']}; border:1px solid {C['border']};"
            f" border-radius:10px; }}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 14)
        lay.setSpacing(12)

        # ── 抓取間隔 ───────────────────────────────────────────────────────
        lay.addWidget(_lbl("抓取間隔", C["gray"], 11))
        int_row = QHBoxLayout()
        int_row.setSpacing(6)
        self._interval_grp = QButtonGroup(self)
        for s in [1, 2, 3, 5, 10, 30]:
            rb = QRadioButton(f"{s}s")
            if s == cfg.get("interval", 1):
                rb.setChecked(True)
            self._interval_grp.addButton(rb, s)
            rb.toggled.connect(lambda chk, v=s: cfg.update({"interval": v}) if chk else None)
            int_row.addWidget(rb)
        int_row.addStretch()
        int_row.addWidget(_lbl("自訂:", C["gray"], 12))
        self._cust = QLineEdit()
        self._cust.setFixedWidth(60)
        self._cust.setPlaceholderText("秒")
        int_row.addWidget(self._cust)
        btn_ap = QPushButton("套用")
        btn_ap.setFixedWidth(56)
        btn_ap.clicked.connect(self._apply_custom)
        int_row.addWidget(btn_ap)
        lay.addLayout(int_row)

        # ── 誤報門檻 ───────────────────────────────────────────────────────
        thr_row = QHBoxLayout()
        thr_row.setSpacing(8)
        thr_row.addWidget(_lbl("誤報門檻", C["gray"], 11))
        self._thresh = QDoubleSpinBox()
        self._thresh.setRange(0.1, 50.0)
        self._thresh.setSingleStep(0.5)
        self._thresh.setValue(cfg.get("threshold", 1.0))
        self._thresh.setFixedWidth(90)
        self._thresh.valueChanged.connect(lambda v: cfg.update({"threshold": v}))
        thr_row.addWidget(self._thresh)
        thr_row.addWidget(_lbl("% 最大經驗 視為誤報門檻", C["gray"], 12))
        thr_row.addStretch()
        lay.addLayout(thr_row)

        lay.addWidget(_sep())

        # ── 顯示區塊 ───────────────────────────────────────────────────────
        lay.addWidget(_lbl("顯示區塊", C["gray"], 11))
        vis_row = QHBoxLayout()
        vis_row.setSpacing(20)
        _vis_map = {
            "stats":     "EXP/s & %/hr",
            "ttl":       "預計升等",
            "chart_pct": "EXP% 趨勢圖",
            "chart_eps": "EXP/s 趨勢圖",
            "log":       "紀錄面板",
        }
        self._vis_cbs: dict[str, QCheckBox] = {}
        for key, label in _vis_map.items():
            cb = QCheckBox(label)
            cb.setChecked(vis.get(key, True))
            cb.toggled.connect(lambda chk, k=key: self._on_vis(k, chk))
            self._vis_cbs[key] = cb
            vis_row.addWidget(cb)
        vis_row.addStretch()
        lay.addLayout(vis_row)

        # 圖表上限
        cmax_row = QHBoxLayout()
        cmax_row.setSpacing(8)
        cmax_row.addWidget(_lbl("圖表上限", C["gray"], 11))
        self._chart_max_spin = QSpinBox()
        self._chart_max_spin.setRange(100, 500000)
        self._chart_max_spin.setSingleStep(3600)
        self._chart_max_spin.setValue(cfg.get("chart_max", 28800))
        self._chart_max_spin.setFixedWidth(100)
        self._chart_max_spin.valueChanged.connect(lambda v: cfg.update({"chart_max": v}))
        cmax_row.addWidget(self._chart_max_spin)
        cmax_row.addWidget(_lbl("筆  (預設 28800 = 8 小時)", C["gray"], 11))
        cmax_row.addStretch()
        lay.addLayout(cmax_row)

    def _on_vis(self, key: str, val: bool):
        self._vis[key] = val
        self.vis_changed.emit()

    def _apply_custom(self):
        try:
            v = int(self._cust.text())
            if v < 1: raise ValueError
            self._cfg["interval"] = v
            checked = self._interval_grp.checkedButton()
            if checked:
                self._interval_grp.setExclusive(False)
                checked.setChecked(False)
                self._interval_grp.setExclusive(True)
        except ValueError:
            pass

    @property
    def threshold(self): return self._thresh.value()


# ══════════════════════════════════════════════════════════════════════════════
# 主視窗
# ══════════════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        set_dpi_awareness()
        self.setWindowTitle("楓之谷 EXP 監控")
        self.setMinimumWidth(680)
        self.setMinimumHeight(480)
        self.setStyleSheet(QSS)

        self._cfg: dict = {"interval": 1, "threshold": 1.0, "chart_max": 28800}
        self._vis: dict = {"stats": True, "ttl": True,
                           "chart_pct": True, "chart_eps": True, "log": True}
        self._thread: QThread | None = None
        self._worker: MonitorWorker | None = None
        self._guard    = DigitGuard()
        self._rate     = RateTracker()
        self._prev_pct:     float | None = None
        self._prev_exp_int: int   | None = None
        self._exp_fail_streak: int = 0
        self._max_exp_est:  float | None = None   # 每等最大EXP估計（由 exp÷pct 推算）

        pg.setConfigOptions(antialias=True)
        self._build_ui()

        self._chart_timer = QTimer(self)
        self._chart_timer.timeout.connect(self._refresh_chart)
        self._chart_timer.start(1000)

    # ── Build UI ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        root_lay = QVBoxLayout(root)
        root_lay.setContentsMargins(0, 0, 0, 0)
        root_lay.setSpacing(0)

        # ─── Header ───────────────────────────────────────────────────────
        hdr = QFrame()
        hdr.setFixedHeight(50)
        hdr.setStyleSheet(
            f"QFrame {{ background:{C['bg2']}; border-bottom:1px solid {C['border']}; }}")
        hdr_lay = QHBoxLayout(hdr)
        hdr_lay.setContentsMargins(16, 0, 12, 0)
        hdr_lay.setSpacing(10)

        title_lb = QLabel("🍁  楓之谷 EXP 監控")
        title_lb.setStyleSheet(
            f"color:{C['white']}; font-size:15px; font-weight:700;"
            f" font-family:'微軟正黑體'; background:transparent; border:none;")
        hdr_lay.addWidget(title_lb)
        hdr_lay.addStretch()

        self._status_lbl = QLabel("待機")
        self._status_lbl.setStyleSheet(
            f"color:{C['gray']}; font-size:12px; background:transparent; border:none;")
        self._status_dot = QLabel("●")
        self._status_dot.setStyleSheet(
            f"color:{C['gray']}; font-size:15px; background:transparent; border:none;")
        hdr_lay.addWidget(self._status_lbl)
        hdr_lay.addWidget(self._status_dot)

        self._btn_cfg = QPushButton("⚙")
        self._btn_cfg.setFixedSize(36, 36)
        self._btn_cfg.setToolTip("設定")
        self._btn_cfg.clicked.connect(self._toggle_settings)
        hdr_lay.addWidget(self._btn_cfg)

        root_lay.addWidget(hdr)

        # ─── Scroll area ──────────────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("border:none;")
        root_lay.addWidget(scroll)

        body = QWidget()
        body.setStyleSheet(f"background:{C['bg']};")
        scroll.setWidget(body)
        self._body_lay = QVBoxLayout(body)
        self._body_lay.setContentsMargins(14, 14, 14, 14)
        self._body_lay.setSpacing(10)

        # ─── Settings panel ───────────────────────────────────────────────
        self._settings = SettingsPanel(self._cfg, self._vis)
        self._settings.setVisible(False)
        self._settings.vis_changed.connect(self._apply_vis)
        self._body_lay.addWidget(self._settings)

        # ─── EXP display card ─────────────────────────────────────────────
        exp_card, exp_lay = _card(16, 14)
        self._body_lay.addWidget(exp_card)

        exp_lay.addWidget(_lbl("EXP 百分比", C["gray"], 11))

        self._pct_lbl = QLabel("—")
        self._pct_lbl.setStyleSheet(
            f"color:{C['cyan']}; font-size:44px; font-weight:700;"
            f" font-family:Consolas; background:transparent;")
        exp_lay.addWidget(self._pct_lbl)

        self._pct_bar = QProgressBar()
        self._pct_bar.setRange(0, 10000)
        self._pct_bar.setValue(0)
        self._pct_bar.setFixedHeight(6)
        self._pct_bar.setTextVisible(False)
        self._pct_bar.setStyleSheet(f"""
            QProgressBar {{
                background: {C['bg3']};
                border: none;
                border-radius: 3px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {C['accent']}, stop:1 {C['cyan']});
                border-radius: 3px;
            }}
        """)
        exp_lay.addWidget(self._pct_bar)

        raw_row = QHBoxLayout()
        raw_row.addWidget(_lbl("原始 EXP", C["gray"], 11))
        raw_row.addStretch()
        self._digit_lbl = _lbl(f"位數學習：學習中 0/{LOCK_REQUIRED}", C["gray"], 11)
        raw_row.addWidget(self._digit_lbl)
        exp_lay.addLayout(raw_row)

        self._exp_lbl = QLabel("—")
        self._exp_lbl.setStyleSheet(
            f"color:{C['white']}; font-size:16px; font-family:Consolas; background:transparent;")
        exp_lay.addWidget(self._exp_lbl)

        thresh_row = QHBoxLayout()
        thresh_row.addWidget(_lbl("動態門檻：", C["gray"], 11))
        self._thresh_lbl = _lbl("—（未建立）", C["gray"], 11, mono=True)
        thresh_row.addWidget(self._thresh_lbl)
        thresh_row.addStretch()
        exp_lay.addLayout(thresh_row)

        # ─── Stats row ────────────────────────────────────────────────────
        self._stats_widget = QWidget()
        self._stats_widget.setStyleSheet("background:transparent;")
        stats_lay = QHBoxLayout(self._stats_widget)
        stats_lay.setContentsMargins(0, 0, 0, 0)
        stats_lay.setSpacing(10)

        self._tile_eps = StatTile("EXP / 秒", "近 1 分鐘", 30)
        self._tile_pph = StatTile("EXP% / 小時", "近 1 小時", 30)
        self._tile_eps.setMinimumHeight(85)
        self._tile_pph.setMinimumHeight(85)
        stats_lay.addWidget(self._tile_eps)
        stats_lay.addWidget(self._tile_pph)
        self._body_lay.addWidget(self._stats_widget)

        # ─── TTL tile ─────────────────────────────────────────────────────
        self._ttl_widget = StatTile("⏱  預計升等時間", "依 %/hr 計算", 44)
        self._ttl_widget.setMinimumHeight(105)
        self._body_lay.addWidget(self._ttl_widget)

        # ─── Chart ────────────────────────────────────────────────────────
        ax_pen = pg.mkPen(C["gray"])

        # 圖表 1：EXP% 趨勢
        self._chart_pct_widget, cpct_lay = _card(12, 10)
        cpct_lay.addWidget(_lbl("EXP % 趨勢", C["gray"], 11))
        self._plot = pg.PlotWidget(background=C["plot_bg"])
        self._plot.setFixedHeight(180)
        self._plot.showGrid(x=True, y=True, alpha=0.12)
        self._plot.setLabel("left",   "EXP %",      color=C["gray"])
        self._plot.setLabel("bottom", "時間 (分鐘)", color=C["gray"])
        self._plot.getAxis("left").setTextPen(ax_pen)
        self._plot.getAxis("bottom").setTextPen(ax_pen)
        self._curve = self._plot.plot(
            pen=pg.mkPen(C["cyan"], width=2),
            symbolBrush=pg.mkBrush(C["cyan"]),
            symbolSize=4, symbol="o")
        cpct_lay.addWidget(self._plot)
        self._body_lay.addWidget(self._chart_pct_widget)

        # 圖表 2：EXP/s 歷史（近60秒平均，每秒採樣）
        self._chart_eps_widget, ceps_lay = _card(12, 10)
        ceps_lay.addWidget(_lbl("EXP / 秒  趨勢（每秒採樣近1分鐘均值）", C["gray"], 11))
        self._plot_eps = pg.PlotWidget(background=C["plot_bg"])
        self._plot_eps.setFixedHeight(160)
        self._plot_eps.showGrid(x=True, y=True, alpha=0.12)
        self._plot_eps.setLabel("left",   "EXP/s",    color=C["gray"])
        self._plot_eps.setLabel("bottom", "時間 (分鐘)", color=C["gray"])
        self._plot_eps.getAxis("left").setTextPen(ax_pen)
        self._plot_eps.getAxis("bottom").setTextPen(ax_pen)
        self._curve_eps = self._plot_eps.plot(
            pen=pg.mkPen(C["green"], width=2))
        ceps_lay.addWidget(self._plot_eps)
        self._body_lay.addWidget(self._chart_eps_widget)

        # ─── Controls ─────────────────────────────────────────────────────
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(8)

        self._btn_start = QPushButton("▶  開始監控")
        self._btn_start.setFixedHeight(38)
        self._btn_start.setStyleSheet(
            f"QPushButton {{ background:#1a7f37; color:{C['white']};"
            f" border:none; border-radius:6px; font-size:13px; font-weight:600; }}"
            f"QPushButton:hover {{ background:#2da44e; }}")
        self._btn_start.clicked.connect(self._start)

        self._btn_stop = QPushButton("■  停止")
        self._btn_stop.setFixedHeight(38)
        self._btn_stop.setEnabled(False)
        self._btn_stop.setStyleSheet(
            f"QPushButton {{ background:#b91c1c; color:{C['white']};"
            f" border:none; border-radius:6px; font-size:13px; font-weight:600; }}"
            f"QPushButton:hover {{ background:#dc2626; }}")
        self._btn_stop.clicked.connect(self._stop)

        btn_clear_log = QPushButton("清除紀錄")
        btn_clear_log.setFixedHeight(38)
        btn_clear_log.clicked.connect(self._clear_log)

        btn_clear_chart = QPushButton("清空圖表")
        btn_clear_chart.setFixedHeight(38)
        btn_clear_chart.clicked.connect(self._clear_charts)

        ctrl_row.addWidget(self._btn_start)
        ctrl_row.addWidget(self._btn_stop)
        ctrl_row.addWidget(btn_clear_log)
        ctrl_row.addWidget(btn_clear_chart)
        ctrl_row.addStretch()
        self._body_lay.addLayout(ctrl_row)

        # ─── Log ──────────────────────────────────────────────────────────
        self._log_widget = QTextEdit()
        self._log_widget.setReadOnly(True)
        self._log_widget.setFixedHeight(180)
        self._body_lay.addWidget(self._log_widget)

        self._body_lay.addStretch()
        self._apply_vis()

    # ── Settings toggle ───────────────────────────────────────────────────────
    def _toggle_settings(self):
        sz    = self.size()
        open_ = not self._settings.isVisible()
        self._settings.setVisible(open_)
        if open_:
            self._btn_cfg.setStyleSheet(
                f"QPushButton {{ background:{C['accent']}; border-color:{C['accent']};"
                f" border-radius:6px; color:{C['white']}; font-size:14px;"
                f" border:1px solid {C['accent']}; }}")
        else:
            self._btn_cfg.setStyleSheet("")
        self.resize(sz)

    def _apply_vis(self):
        sz = self.size()
        self._stats_widget.setVisible(self._vis.get("stats", True))
        self._ttl_widget.setVisible(self._vis.get("ttl", True))
        self._chart_pct_widget.setVisible(self._vis.get("chart_pct", True))
        self._chart_eps_widget.setVisible(self._vis.get("chart_eps", True))
        self._log_widget.setVisible(self._vis.get("log", True))
        self.resize(sz)

    # ── Control ───────────────────────────────────────────────────────────────
    def _start(self):
        if self._thread and self._thread.isRunning(): return
        self._prev_pct        = None
        self._prev_exp_int    = None
        self._exp_fail_streak = 0
        self._max_exp_est     = None
        self._guard.reset()
        self._rate.reset()
        self._digit_lbl.setText(f"位數學習：學習中 0/{LOCK_REQUIRED}")
        self._clear_charts()
        for t in [self._tile_eps, self._tile_pph, self._ttl_widget]:
            t.set_value("—")

        self._worker = MonitorWorker(self._cfg)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.start_work)
        self._worker.reading.connect(self._on_reading)
        self._worker.no_window.connect(lambda: self._log_warn("找不到 MapleStory 視窗"))
        self._worker.cap_fail.connect(lambda m: self._log_err(f"截圖失敗：{m}"))
        self._worker.ocr_fail.connect(lambda ts: self._log_warn(f"[{ts}] OCR 無結果"))
        self._worker.status.connect(self._log_info)
        self._worker.error_sig.connect(self._log_err)
        self._thread.start()

        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._set_status("監控中", C["green"])
        self._log_info("監控已啟動")

    def _stop(self):
        if self._worker: self._worker.stop()
        if self._thread:
            self._thread.quit()
            self._thread.wait(3000)
        self._thread = self._worker = None
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._set_status("已停止", C["gray"])
        self._log_info("監控已停止")

    def _clear_log(self):
        self._log_widget.clear()

    # ── 顯示門檻資訊 ─────────────────────────────────────────────────────────
    def _update_thresh_display(self):
        thresh  = self._cfg.get("threshold", 5.0)
        max_exp = self._max_exp_est
        if max_exp is not None and max_exp > 0:
            tol = thresh / 100.0 * max_exp
            t_str = f"{max_exp/1e12:.2f}T" if max_exp >= 1e12 else f"{max_exp/1e9:.2f}B"
            r_str = f"±{tol/1e12:.2f}T"  if tol  >= 1e12 else f"±{tol/1e9:.2f}B"
            self._thresh_lbl.setText(
                f"基準 {t_str}，容差 {thresh:.1f}% ({r_str} EXP)")
            self._thresh_lbl.setStyleSheet(
                f"color:{C['cyan']}; font-size:11px; background:transparent;"
                f" border:none; font-family:Consolas;")
        else:
            self._thresh_lbl.setText("— (max_exp 未建立，首筆讀值建立基準)")
            self._thresh_lbl.setStyleSheet(
                f"color:{C['gray']}; font-size:11px; background:transparent; border:none;")

    # ── Reading handler ───────────────────────────────────────────────────────
    @pyqtSlot(dict)
    def _on_reading(self, d: dict):
        ts  = d["ts"]
        pct = d["pct"]
        exp = d["exp"]
        cap = d["cap"]
        try:
            pct_f = float(pct)
        except Exception:
            return

        # ── EXP 整數：永遠從 max_exp × pct 推算，不依賴 OCR ──────────────
        # 原因：EXP 數字橫跨填充條邊界，部分字元不可見，OCR 必然丟失前幾位
        # max_exp 建立後，pct × max_exp 的精度足夠（誤差 < 0.001%）
        exp_int = None
        if self._max_exp_est is not None and pct_f > 1.0:
            exp_int = round(self._max_exp_est * pct_f / 100.0)
            exp     = f"{exp_int:,}"

        thresh = self._cfg.get("threshold", 5.0)

        # 計算 cur_max：第一次建立基準時用 OCR 讀到的 EXP（若有）
        cur_max = None
        if self._max_exp_est is None and exp:
            # 尚未建立 max_exp_est，嘗試用 OCR 的 EXP 整數建立基準
            try:
                ocr_exp = int(exp.replace(",", ""))
                if ocr_exp > 1_000_000_000:   # 至少 10 億才合理
                    cur_max = ocr_exp / (pct_f / 100.0)
            except Exception:
                pass
        elif self._max_exp_est is not None and exp_int is not None:
            cur_max = exp_int / (pct_f / 100.0)

        self._update_thresh_display()

        # ── 1. max_exp 一致性（核心層）──────────────────────────────────
        # cur_max 應與歷史基準 _max_exp_est 一致；
        # 偏差 > threshold% 代表 exp 或 pct 至少有一個讀錯
        ok1, reason1 = True, ""
        is_lu = pct_f < 5.0 and self._prev_pct is not None and self._prev_pct > 90.0
        if cur_max is not None and self._max_exp_est is not None and not is_lu:
            dev = abs(cur_max - self._max_exp_est) / self._max_exp_est
            if dev > thresh / 100.0:
                ok1     = False
                c_str   = f"{cur_max/1e12:.2f}T" if cur_max >= 1e12 else f"{cur_max/1e9:.2f}B"
                b_str   = f"{self._max_exp_est/1e12:.2f}T" if self._max_exp_est >= 1e12 else f"{self._max_exp_est/1e9:.2f}B"
                reason1 = f"max_exp 偏差 {dev*100:.1f}%  ({c_str} vs 基準{b_str})"

        # ── 2. EXP 只增不減 + 自動重設地板 ──────────────────────────────
        EXP_FLOOR_RESET = 3
        ok2, reason2 = True, ""
        if exp_int is not None and self._prev_exp_int is not None:
            if not is_lu and exp_int < self._prev_exp_int:
                self._exp_fail_streak += 1
                if self._exp_fail_streak >= EXP_FLOOR_RESET:
                    self._log_colored(
                        f"[{ts}]  [RESET] EXP floor reset: "
                        f"{self._prev_exp_int:,} -> {exp_int:,} "
                        f"(streak {self._exp_fail_streak})",
                        C["purple"])
                    self._prev_exp_int    = None
                    self._exp_fail_streak = 0
                else:
                    ok2     = False
                    reason2 = (f"EXP 減少 {self._prev_exp_int - exp_int:,}"
                               f" ({self._exp_fail_streak}/{EXP_FLOOR_RESET})")
            else:
                self._exp_fail_streak = 0

        # ── 3. 位數守衛（輔助）──────────────────────────────────────────
        locked_before = self._guard.locked
        ok3, reason3  = self._guard.check(exp)
        self._update_digit_lbl(locked_before)

        valid  = ok1 and ok2 and ok3
        reason = reason1 or reason2 or reason3

        if valid:
            is_lu = pct_f < 5.0 and self._prev_pct is not None and self._prev_pct > 90.0
            if is_lu:
                self._rate.level_up_reset()
                self._max_exp_est = None   # 新等級重新估算
                self._log_colored(f"[{ts}]  🎉 升等！", C["yellow"])

            self._pct_lbl.setText(f"{pct}%")
            self._pct_lbl.setStyleSheet(
                f"color:{C['cyan']}; font-size:44px; font-weight:700;"
                f" font-family:Consolas; background:transparent;")
            self._pct_bar.setValue(int(pct_f * 100))
            self._exp_lbl.setText(exp or "—")
            self._set_status("監控中", C["green"])

            if exp_int is not None:
                self._rate.add(exp_int, pct_f)
                # 用通過驗證的 cur_max 更新基準（EMA）
                if cur_max is not None:
                    if self._max_exp_est is None:
                        self._max_exp_est = cur_max
                    else:
                        self._max_exp_est = self._max_exp_est * 0.9 + cur_max * 0.1
                self._update_stats(pct_f)

            diff_str = ""
            if self._prev_pct is not None:
                diff_str = f"  ({pct_f - self._prev_pct:+.3f}%)"

            line = f"[{ts}]  {pct}%"
            if exp:     line += f"  {exp}"
            if diff_str: line += diff_str
            line += f"  [{cap}]"
            self._log_colored(line, C["green"])

            self._prev_pct = pct_f
            if exp_int is not None:
                self._prev_exp_int = exp_int
        else:
            self._pct_lbl.setStyleSheet(
                f"color:{C['yellow']}; font-size:44px; font-weight:700;"
                f" font-family:Consolas; background:transparent;")
            self._set_status(f"誤報：{reason}", C["yellow"])
            self._log_colored(
                f"[{ts}]  ⚠ 誤報  pct={pct}%"
                + (f"  exp={exp}" if exp else "")
                + f"  原因：{reason}",
                C["yellow"])

    def _update_digit_lbl(self, locked_before):
        locked = self._guard.locked
        if locked is None:
            self._digit_lbl.setText(
                f"位數學習：學習中 {self._guard.progress}/{LOCK_REQUIRED}")
            self._digit_lbl.setStyleSheet(
                f"color:{C['gray']}; font-size:11px; background:transparent; border:none;")
        else:
            self._digit_lbl.setText(f"位數學習：已鎖定 {locked} 位")
            self._digit_lbl.setStyleSheet(
                f"color:{C['green']}; font-size:11px; background:transparent; border:none;")
            if locked_before != locked:
                ts = datetime.now().strftime("%H:%M:%S")
                self._log_colored(
                    f"[{ts}]  [LOCK] EXP digit locked: {locked}", C["purple"])

    def _update_stats(self, current_pct: float):
        eps = self._rate.exp_per_sec
        pph = self._rate.pct_per_hour
        ttl = self._rate.time_to_level(current_pct)

        if eps is not None:
            self._tile_eps.set_value(
                f"{eps:,.0f}" if eps >= 0 else "—",
                C["cyan"] if eps >= 0 else C["red"])
        if pph is not None:
            self._tile_pph.set_value(
                f"{pph:.3f}",
                C["green"] if pph >= 0 else C["red"])
        self._ttl_widget.set_value(ttl, C["yellow"])

    # ── Chart ─────────────────────────────────────────────────────────────────
    def _clear_charts(self):
        self._curve.setData([], [])
        self._curve_eps.setData([], [])
        self._rate.clear_chart_data()

    def _refresh_chart(self):
        max_pts = self._cfg.get("chart_max", 28800)
        # EXP% 趨勢
        xs, ys = self._rate.chart_data(max_pts)
        if len(xs) >= 2:
            self._curve.setData(xs, ys)
        # EXP/s 歷史（每秒採樣當前近1分鐘均值）
        eps_now = self._rate.exp_per_sec
        if eps_now is not None and eps_now >= 0:
            self._rate.add_eps_sample(eps_now)
        exs, eys = self._rate.eps_sample_data(max_pts)
        if len(exs) >= 2:
            self._curve_eps.setData(exs, eys)

    # ── Status / Log ──────────────────────────────────────────────────────────
    def _set_status(self, text: str, color: str):
        self._status_lbl.setStyleSheet(
            f"color:{color}; font-size:12px; background:transparent; border:none;")
        self._status_lbl.setText(text)
        self._status_dot.setStyleSheet(
            f"color:{color}; font-size:15px; background:transparent; border:none;")

    def _log_info(self, msg: str):
        self._log_colored(f"i  {msg}", C["gray"])

    def _log_warn(self, msg: str):
        self._log_colored(f"!  {msg}", C["yellow"])

    def _log_err(self, msg: str):
        self._log_colored(f"x  {msg}", C["red"])

    def _log_colored(self, msg: str, color: str):
        ts  = datetime.now().strftime("%H:%M:%S")
        cur = self._log_widget.textCursor()
        cur.movePosition(QTextCursor.End)
        fmt = cur.charFormat()
        fmt.setForeground(QColor(color))
        cur.setCharFormat(fmt)
        prefix = "" if msg.startswith("[") else f"[{ts}]  "
        cur.insertText(prefix + msg + chr(10))
        self._log_widget.setTextCursor(cur)
        self._log_widget.ensureCursorVisible()

    def closeEvent(self, event):
        self._stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
