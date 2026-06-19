"""
MapleStory EXP Monitor — Qt UI v2
==================================
pip install PyQt5 pyqtgraph numpy
python exp_monitor_qt.py
"""

import os, sys, time, subprocess, json
from collections import deque
from datetime import datetime

import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QDoubleSpinBox, QTextEdit,
    QButtonGroup, QRadioButton, QLineEdit, QFrame,
    QCheckBox, QScrollArea, QProgressBar, QSizePolicy, QSpinBox, QMessageBox, QFileDialog, QDialog, QDialogButtonBox,
)
from PyQt5.QtCore import Qt, QThread, QObject, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtGui import QColor, QTextCursor, QFont, QImage, QPixmap
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

# 設定存檔位置（可寫入）
def _app_dir():
    """程式所在資料夾：打包後＝exe 旁邊；原始碼＝本檔資料夾。"""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

_CFG_DIR = _app_dir()
CONFIG_PATH = os.path.join(_CFG_DIR, 'config.json')

try:
    from exp_template_ocr import (TemplateOCR, build_mask, _imwrite_u,
                                  build_user_templates, USER_TEMPLATE_DIR)
    _HAS_TEMPLATE = True
except Exception:
    _HAS_TEMPLATE = False

# ══════════════════════════════════════════════════════════════════════════════
# 主題定義
# ══════════════════════════════════════════════════════════════════════════════
THEMES = {
    "dark": {
        # GitHub Dark 原始配色
        "bg":          "#0d1117",
        "bg2":         "#161b22",
        "bg3":         "#21262d",
        "bg_hero":     "#0d1117",
        "border":      "#30363d",
        "border_ui":   "#30363d",
        "accent":      "#e3b341",
        "cyan":        "#58a6ff",
        "green":       "#3fb950",
        "yellow":      "#e3b341",
        "red":         "#f85149",
        "gray":        "#8b949e",
        "white":       "#e6edf3",
        "white_hero":  "#e6edf3",
        "purple":      "#bc8cff",
        "hdr_top":     "#161b22",
        "hdr_bot":     "#0d1117",
        "plot_bg":     "#0d1117",
        "chart_text":  "#8b949e",  # 圖表軸文字（深底→淺字）
        "bg_stat":     "#21262d",  # StatTile bg（HP/MP/STR 感覺）
        "stat_text":   "#e6edf3",  # StatTile 數值文字
        "bg_attr":     "#161b22",  # 偷懶/圖表 panel bg（攻擊力/傷害 感覺）
    },
    "maplestory": {
        # 楓之谷「屬性」面板像素取樣
        "bg":          "#76828D",
        "bg2":         "#86939F",
        "bg3":         "#9CA7B1",
        "bg_hero":     "#3E6279",
        "border":      "#5A6470",
        "border_ui":   "#6E7A88",
        "accent":      "#B78633",
        "cyan":        "#D9D9BC",
        "green":       "#3fb950",
        "yellow":      "#B78633",
        "red":         "#e04040",
        "gray":        "#AABBC6",  # 淺色，在 bg_attr/bg_hero/header 上皆可讀
        "white":       "#D7DCE0",
        "white_hero":  "#D9D9BC",
        "purple":      "#c080f0",
        "hdr_top":     "#5A6470",
        "hdr_bot":     "#3A4550",
        "plot_bg":     "#6C7785",  # 配合 bg_attr，圖表貼齊底板
        "chart_text":  "#1e2028",  # 圖表軸文字（淺底→深字）
        "bg_stat":     "#7F8C98",  # StatTile bg（HP/MP/STR/INT 底色，像素取樣）
        "stat_text":   "#E8EDF1",  # StatTile 數值文字（HP 數字色）
        "bg_attr":     "#6C7785",  # 偷懶/圖表 panel bg（攻擊力/傷害 底色，像素取樣）
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# 色彩（由主題動態生成）
# ══════════════════════════════════════════════════════════════════════════════
C = dict(THEMES["maplestory"])  # 預設楓之谷主題


def _build_qss():
    return f"""
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
    border-radius: 4px;
    padding: 6px 14px;
    font-size: 13px;
    font-weight: 700;
    border: 1px solid {C['border']};
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #3a4254, stop:1 #2f3543);
    color: {C['white']};
}}
QPushButton:hover {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #4a5264, stop:1 #3f4553);
    border-color: {C['accent']};
    color: #ffffff;
}}
QPushButton:pressed {{
    background: #2f3543;
    border-color: {C['accent']};
}}
QPushButton:disabled {{
    color: #7a8898;
    border-color: {C['border']};
    background: #2f3543;
}}
QDoubleSpinBox, QLineEdit, QSpinBox {{
    background: #2f3543;
    border: 1px solid {C['border']};
    border-radius: 4px;
    padding: 4px 8px;
    color: {C['white']};
    font-family: Consolas;
    font-size: 13px;
}}
QDoubleSpinBox:focus, QLineEdit:focus, QSpinBox:focus {{
    border-color: {C['accent']};
}}
QTextEdit {{
    background: {C['bg_attr']};
    border: 1px solid {C['border']};
    border-radius: 4px;
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
    border: 2px solid {C['border']};
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
    border-radius: 3px;
    border: 2px solid {C['border']};
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
/* ── 主題色標籤（切換主題時 re-polish 刷新）─────────────────── */
QLabel[lbl_clr="gray"]      {{ color: {C['gray']};      }}
QLabel[lbl_clr="stat_text"] {{ color: {C['stat_text']}; }}
QLabel[lbl_clr="white"]  {{ color: {C['white']};  }}
QLabel[lbl_clr="cyan"]   {{ color: {C['cyan']};   }}
QLabel[lbl_clr="green"]  {{ color: {C['green']};  }}
QLabel[lbl_clr="red"]    {{ color: {C['red']};    }}
QLabel[lbl_clr="accent"] {{ color: {C['accent']}; }}
QLabel[lbl_clr="purple"] {{ color: {C['purple']}; }}
/* ── 分隔線 ─────────────────────────────────────────────────── */
QFrame[sep_line="true"]  {{ background: {C['border']}; border:none; }}

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
        return ([t for t, _, _ in pts],
                [p for _, _, p in pts])

    def add_eps_sample(self, eps: float):
        """每秒由 chart timer 呼叫，記錄當前 EXP/s 樣本。"""
        self._eps_samples.append((time.time(), eps))

    def eps_sample_data(self, max_pts: int = 28800) -> tuple[list, list]:
        """EXP/s 歷史（以分鐘為 X 軸）。"""
        pts = list(self._eps_samples)[-max_pts:]
        if len(pts) < 2: return [], []
        return ([t for t, _ in pts],
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
        self._running = True
        try:
            self._ocr = None
            self._use_tess = False
            self.status.emit("初始化辨識器…")

            # 1) 優先：模板形狀比對
            if _HAS_TEMPLATE:
                try:
                    o = TemplateOCR()
                    if o.is_ready():
                        self._ocr = o
                        self.status.emit("OCR=Template(形狀比對) 就緒")
                    else:
                        self.status.emit(
                            f"模板未就緒（只載入到 {len(o.templates)} 個；"
                            f"路徑 {o.template_dir}）")
                except Exception as e:
                    self.error_sig.emit(f"模板辨識器載入失敗：{e!r}")
            else:
                self.status.emit("模板模組未載入（_HAS_TEMPLATE=False）")

            # 2) 退回：Tesseract → EasyOCR（兩者都包了 try，缺了也不會無聲當掉）
            if self._ocr is None:
                try:
                    self._use_tess = _setup_tess()
                except Exception as e:
                    self.error_sig.emit(f"Tesseract 偵測失敗：{e!r}")
                    self._use_tess = False
                if self._use_tess:
                    self.status.emit("OCR=Tesseract 就緒")
                else:
                    try:
                        _init_easy()
                        self.status.emit("OCR=EasyOCR 就緒")
                    except Exception as e:
                        self.error_sig.emit(
                            "找不到可用的 OCR 引擎："
                            "模板未就緒、未安裝 Tesseract、EasyOCR 也不可用"
                            f"（{e!r}）。請確認 templates 資料夾有打包進來。")
                        self._fatal_startup("no_ocr_engine")
                        self._running = False
                        self.status.emit("已停止：沒有可用的 OCR 引擎")
                        return
        except Exception as e:
            self._fatal_startup(repr(e))
            self.error_sig.emit(f"啟動失敗：{e!r}")
            self._running = False
            return

        self._loop()

    def _fatal_startup(self, why):
        """把啟動失敗的完整 traceback 寫到使用者家目錄，方便回報。"""
        import traceback, os as _os
        try:
            path = _os.path.join(_CFG_DIR, "EXPMonitor_error.log")
            with open(path, "a", encoding="utf-8") as f:
                f.write("=" * 60 + chr(10))
                f.write(f"start_work fatal: {why}" + chr(10))
                f.write(traceback.format_exc() + chr(10))
            self.error_sig.emit(f"（錯誤詳情已寫到 {path}）")
        except Exception:
            pass

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
            r = self._ocr.recognize_row(text_band, expected_digits=self.cfg.get("exp_digits", 0))
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
def _lbl(text, color="white", size=13, bold=False, mono=False):
    """
    color = C dict key（如 "gray", "white", "cyan"）
    顏色由 QSS QLabel[lbl_clr=...] 控制，切換主題後 re-polish 即可刷新。
    """
    lb = QLabel(text)
    lb.setProperty("lbl_clr", color)
    w  = "600" if bold else "normal"
    ff = "Consolas" if mono else "'微軟正黑體','Segoe UI',sans-serif"
    # 只寫排版，不寫 color（由 QSS 管）
    lb.setStyleSheet(
        f"font-size:{size}px; font-weight:{w};"
        f" font-family:{ff}; background:transparent; border:none;")
    return lb


def _sep():
    line = QFrame()
    line.setFixedHeight(1)
    line.setProperty("sep_line", True)
    line.setStyleSheet("border:none;")  # 顏色由 QSS QFrame[sep_line=true] 管
    return line


def _card(pad_h=16, pad_v=14):
    f = QFrame()
    f.setStyleSheet(
        f"QFrame {{ background:{C['bg2']}; border:1px solid {C['border_ui']};"
        f" border-radius:6px; }}")
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
            f"QFrame {{ background:{C['bg_stat']}; border:1px solid {C['border_ui']};"
            f" border-radius:6px; }}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 12)
        lay.setSpacing(2)

        title_row = QHBoxLayout()
        title_row.setSpacing(6)
        self._title_lb = _lbl(title, "white", 12, bold=True)
        title_row.addWidget(self._title_lb)
        if subtitle:
            title_row.addWidget(_lbl(subtitle, "gray", 10))
        title_row.addStretch()
        lay.addLayout(title_row)

        self._val_lb = _lbl("—", "stat_text", value_size, bold=True, mono=True)
        lay.addWidget(self._val_lb)

    def set_value(self, text: str, color: str = None):
        color = color or C["stat_text"]
        self._val_lb.setText(text)
        self._val_lb.setStyleSheet(
            f"color:{color}; font-size:{self._vsize}px; font-weight:600;"
            f" font-family:Consolas; background:transparent; border:none;")

    def refresh_theme(self):
        """主題切換後重新套用樣式"""
        self.setStyleSheet(
            f"QFrame {{ background:{C['bg_stat']}; border:1px solid {C['border_ui']};"
            f" border-radius:6px; }}")
        # 重置 value label 顏色（保持目前文字）
        cur_text = self._val_lb.text()
        self._val_lb.setStyleSheet(
            f"color:{C['cyan']}; font-size:{self._vsize}px; font-weight:600;"
            f" font-family:Consolas; background:transparent; border:none;")


# ══════════════════════════════════════════════════════════════════════════════
# SettingsPanel — 可折疊設定面板
# ══════════════════════════════════════════════════════════════════════════════
class SettingsPanel(QFrame):
    vis_changed   = pyqtSignal()
    slack_test    = pyqtSignal()
    theme_changed = pyqtSignal(str)   # 傳出 "dark" / "maplestory"

    def __init__(self, cfg: dict, vis: dict, parent=None):
        super().__init__(parent)
        self._cfg = cfg
        self._vis = vis
        self.setStyleSheet(
            f"QFrame {{ background:{C['bg_attr']}; border:1px solid {C['border_ui']};"
            f" border-radius:6px; }}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 14)
        lay.setSpacing(12)

        # ── 抓取間隔 ───────────────────────────────────────────────────────
        lay.addWidget(_lbl("抓取間隔", "gray", 11))
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
        int_row.addWidget(_lbl("自訂:", "gray", 12))
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
        thr_row.addWidget(_lbl("誤報門檻", "gray", 11))
        self._thresh = QDoubleSpinBox()
        self._thresh.setRange(0.1, 50.0)
        self._thresh.setSingleStep(0.5)
        self._thresh.setValue(cfg.get("threshold", 1.0))
        self._thresh.setFixedWidth(90)
        self._thresh.valueChanged.connect(lambda v: cfg.update({"threshold": v}))
        thr_row.addWidget(self._thresh)
        thr_row.addWidget(_lbl("% 最大經驗 視為誤報門檻", "gray", 12))
        thr_row.addStretch()
        lay.addLayout(thr_row)

        # ── 經驗位數（輔助辨識）────────────────────────────────────────────
        dig_row = QHBoxLayout()
        dig_row.setSpacing(8)
        dig_row.addWidget(_lbl("經驗位數", "gray", 11))
        self._expdig = QSpinBox()
        self._expdig.setRange(0, 20)
        self._expdig.setValue(cfg.get("exp_digits", 0))
        self._expdig.setFixedWidth(90)
        self._expdig.valueChanged.connect(lambda v: cfg.update({"exp_digits": v}))
        dig_row.addWidget(self._expdig)
        dig_row.addWidget(_lbl("位（0=自動）；剔除填充邊界誤判的多餘數字", "gray", 12))
        dig_row.addStretch()
        lay.addLayout(dig_row)

        lay.addWidget(_sep())

        # ── 顯示區塊 ───────────────────────────────────────────────────────
        lay.addWidget(_lbl("顯示區塊", "gray", 11))
        vis_row = QHBoxLayout()
        vis_row.setSpacing(20)
        _vis_map = {
            "stats":     "EXP/s & %/hr",
            "ttl":       "預計升等",
            "slack":     "偷懶偵測",
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
        cmax_row.addWidget(_lbl("圖表上限", "gray", 11))
        self._chart_max_spin = QSpinBox()
        self._chart_max_spin.setRange(100, 500000)
        self._chart_max_spin.setSingleStep(3600)
        self._chart_max_spin.setValue(cfg.get("chart_max", 28800))
        self._chart_max_spin.setFixedWidth(100)
        self._chart_max_spin.valueChanged.connect(lambda v: cfg.update({"chart_max": v}))
        cmax_row.addWidget(self._chart_max_spin)
        cmax_row.addWidget(_lbl("筆  (預設 28800 = 8 小時)", "gray", 11))
        cmax_row.addStretch()
        lay.addLayout(cmax_row)

        lay.addWidget(_sep())
        lay.addWidget(_lbl("偷懶偵測動作", "gray", 11))

        smsg_row = QHBoxLayout(); smsg_row.setSpacing(8)
        smsg_row.addWidget(_lbl("警告文字", "gray", 12))
        self._slack_msg = QLineEdit(cfg.get("slack_msg", "！偷懶警告！EXP 已 {sec} 秒沒有增加"))
        self._slack_msg.setPlaceholderText("可用 {sec} {exp} {pct} 代入數值")
        self._slack_msg.textChanged.connect(lambda t: cfg.update({"slack_msg": t}))
        smsg_row.addWidget(self._slack_msg)
        lay.addLayout(smsg_row)

        scmd_row = QHBoxLayout(); scmd_row.setSpacing(8)
        scmd_row.addWidget(_lbl("觸發腳本", "gray", 12))
        self._slack_cmd = QLineEdit(cfg.get("slack_cmd", ""))
        self._slack_cmd.setPlaceholderText("例：python hook.py（留空=只跳視窗，不執行）")
        self._slack_cmd.textChanged.connect(lambda t: cfg.update({"slack_cmd": t}))
        scmd_row.addWidget(self._slack_cmd)
        _bb = QPushButton("瀏覽"); _bb.setFixedWidth(56)
        _bb.clicked.connect(self._slack_browse_setting); scmd_row.addWidget(_bb)
        _bt = QPushButton("測試"); _bt.setFixedWidth(56)
        _bt.clicked.connect(self.slack_test.emit); scmd_row.addWidget(_bt)
        lay.addLayout(scmd_row)

        lay.addWidget(_sep())
        lay.addWidget(_lbl("效率過低提醒", "gray", 11))
        le_row = QHBoxLayout(); le_row.setSpacing(8)
        self._loweff_enable = QCheckBox("啟用")
        self._loweff_enable.setChecked(bool(cfg.get("loweff_enable", False)))
        self._loweff_enable.toggled.connect(lambda c: cfg.update({"loweff_enable": c}))
        le_row.addWidget(self._loweff_enable)
        le_row.addWidget(_lbl("EXP/s 低於", "gray", 12))
        self._loweff_thr = QLineEdit(str(int(cfg.get("loweff_threshold", 0) or 0)))
        self._loweff_thr.setFixedWidth(150)
        self._loweff_thr.setPlaceholderText("例：300000000")
        self._loweff_thr.textChanged.connect(self._apply_loweff_thr)
        le_row.addWidget(self._loweff_thr)
        le_row.addWidget(_lbl("時，於畫面顯示警告", "gray", 12))
        le_row.addStretch()
        lay.addLayout(le_row)

        lay.addWidget(_sep())
        # ── 介面主題 ────────────────────────────────────────────────────────
        lay.addWidget(_lbl("介面主題", "gray", 11))
        theme_row = QHBoxLayout()
        theme_row.setSpacing(16)
        self._theme_grp = QButtonGroup(self)
        for tid, tlabel in [("dark", "🌑  原始深色"), ("maplestory", "🍁  楓之谷")]:
            rb = QRadioButton(tlabel)
            if tid == cfg.get("theme", "maplestory"):
                rb.setChecked(True)
            self._theme_grp.addButton(rb)
            rb.toggled.connect(lambda chk, t=tid: self.theme_changed.emit(t) if chk else None)
            theme_row.addWidget(rb)
        theme_row.addStretch()
        lay.addLayout(theme_row)

    def _apply_loweff_thr(self, t):
        try:
            self._cfg["loweff_threshold"] = float(t.replace(",", "").strip() or 0)
        except ValueError:
            pass

    def _slack_browse_setting(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "選擇觸發腳本", "",
            "可執行檔 (*.py *.bat *.cmd *.exe);;所有檔案 (*.*)")
        if path:
            cmd = f'python "{path}"' if path.lower().endswith(".py") else f'"{path}"'
            self._slack_cmd.setText(cmd)

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
        self.setStyleSheet(_build_qss())

        self._cfg: dict = {"interval": 1, "threshold": 1.0, "chart_max": 28800,
                           "slack_msg": "！偷懶警告！EXP 已 {sec} 秒沒有增加", "slack_cmd": "",
                           "loweff_enable": False, "loweff_threshold": 0}
        self._vis: dict = {"stats": True, "ttl": True,
                           "chart_pct": True, "chart_eps": True, "log": True}
        self._load_config()   # 套用上次存檔的設定（在建立 UI 前）
        self._thread: QThread | None = None
        self._worker: MonitorWorker | None = None
        self._guard    = DigitGuard()
        self._rate     = RateTracker()
        self._prev_pct:     float | None = None
        self._prev_exp_int: int   | None = None
        self._exp_fail_streak: int = 0
        self._max_exp_est:  float | None = None   # 每等最大EXP估計（由 exp÷pct 推算）
        # 偷懶偵測狀態
        self._slack_last_exp: int | None = None
        self._slack_streak = 0
        self._slack_t0: float | None = None
        self._slack_triggered = False
        self._slack_alert = None
        self._loweff_alert = None
        # 本次監控統計
        self._sess_start_ts = None
        self._sess_start_pct = None
        self._sess_prev_pct = None
        self._sess_levels = 0

        # 套用儲存的主題（在 _load_config 之後）
        _saved_theme = self._cfg.get("theme", "maplestory")
        if _saved_theme in THEMES:
            C.update(THEMES[_saved_theme])

        pg.setConfigOptions(antialias=True, foreground=C["chart_text"])
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
        self._hdr = hdr = QFrame()
        hdr.setFixedHeight(50)
        hdr.setStyleSheet(
            f"QFrame {{ background: qlineargradient(x1:0, y1:0, x2:0, y2:1,"
            f" stop:0 {C['hdr_top']}, stop:1 {C['hdr_bot']});"
            f" border-bottom: 1px solid {C['border']}; }}")
        hdr_lay = QHBoxLayout(hdr)
        hdr_lay.setContentsMargins(16, 0, 12, 0)
        hdr_lay.setSpacing(10)

        self._title_lb = title_lb = QLabel("🍁  楓之谷 EXP 監控")
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

        self._body_widget = body = QWidget()
        body.setStyleSheet(f"background:{C['bg']};")
        scroll.setWidget(body)
        self._body_lay = QVBoxLayout(body)
        self._body_lay.setContentsMargins(14, 14, 14, 14)
        self._body_lay.setSpacing(10)

        # ─── Settings panel ───────────────────────────────────────────────
        self._settings = SettingsPanel(self._cfg, self._vis)
        self._settings.setVisible(False)
        self._settings.vis_changed.connect(self._apply_vis)
        self._settings.slack_test.connect(self._slack_test)
        self._settings.theme_changed.connect(self._apply_theme)
        self._body_lay.addWidget(self._settings)

        # 警示橫幅（顯示在視窗內，OBS 視窗擷取抓得到；取代彈窗）。兩種警告各一條，獨立顯示。
        def _mk_banner(bg):
            lb = QLabel(""); lb.setVisible(False); lb.setWordWrap(True)
            lb.setAlignment(Qt.AlignCenter)
            lb.setStyleSheet(
                f"background:{bg}; color:#ffffff; font-size:20px; font-weight:800;"
                f" font-family:'微軟正黑體'; border-radius:8px; padding:10px;")
            return lb
        self._slack_banner  = _mk_banner(C["red"])      # 偷懶警告（紅）
        self._loweff_banner = _mk_banner("#b8860b")     # 效率過低（暗金/橘）
        self._body_lay.addWidget(self._slack_banner)
        self._body_lay.addWidget(self._loweff_banner)

        # ─── EXP display card（戰鬥力列樣式：最深底色 + 純白大字）──────────
        self._exp_card, exp_lay = _card(16, 14)
        exp_card = self._exp_card  # alias
        # 覆寫為「戰鬥力列」配色
        exp_card.setStyleSheet(
            f"QFrame {{ background:{C['bg_hero']}; border:1px solid {C['border_ui']};"
            f" border-radius:6px; }}")
        self._body_lay.addWidget(exp_card)

        exp_lay.addWidget(_lbl("EXP 百分比", "gray", 11))

        self._pct_lbl = QLabel("—")
        self._pct_lbl.setStyleSheet(
            f"color:{C['white_hero']}; font-size:44px; font-weight:700;"
            f" font-family:Consolas; background:transparent;")
        exp_lay.addWidget(self._pct_lbl)

        self._pct_bar = QProgressBar()
        self._pct_bar.setRange(0, 10000)
        self._pct_bar.setValue(0)
        self._pct_bar.setFixedHeight(10)
        self._pct_bar.setTextVisible(False)
        self._pct_bar.setStyleSheet(f"""
            QProgressBar {{
                background: {C['bg']};
                border: 1px solid {C['border_ui']};
                border-radius: 4px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f8c840, stop:1 #d08010);
                border-radius: 3px;
            }}
        """)
        exp_lay.addWidget(self._pct_bar)

        raw_row = QHBoxLayout()
        raw_row.addWidget(_lbl("EXP", "gray", 11))
        raw_row.addStretch()
        self._digit_lbl = _lbl(f"位數學習：學習中 0/{LOCK_REQUIRED}", "gray", 11)
        raw_row.addWidget(self._digit_lbl)
        exp_lay.addLayout(raw_row)

        self._exp_lbl = QLabel("—")
        self._exp_lbl.setStyleSheet(
            f"color:{C['white']}; font-size:16px; font-family:Consolas; background:transparent;")
        exp_lay.addWidget(self._exp_lbl)

        thresh_row = QHBoxLayout()
        thresh_row.addWidget(_lbl("升級所需：", "gray", 11))
        self._thresh_lbl = _lbl("—（未建立）", "gray", 11, mono=True)
        thresh_row.addWidget(self._thresh_lbl)
        thresh_row.addStretch()
        exp_lay.addLayout(thresh_row)

        sess_row = QHBoxLayout(); sess_row.setSpacing(16)
        self._sess_start_lbl = _lbl("起始 —", "gray", 12, mono=True)
        self._sess_dur_lbl   = _lbl("持續 00:00:00", "gray", 12, mono=True)
        self._sess_gain_lbl  = _lbl("增加 +0.000%", "green", 14, mono=True, bold=True)
        sess_row.addWidget(self._sess_start_lbl)
        sess_row.addWidget(self._sess_dur_lbl)
        sess_row.addWidget(self._sess_gain_lbl)
        sess_row.addStretch()
        exp_lay.addLayout(sess_row)

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

        # ─── 偷懶偵測 card ─────────────────────────────────────────────────
        self._slack_card, slack_lay = _card(16, 14)
        self._slack_card.setStyleSheet(
            f"QFrame {{ background:{C['bg_attr']}; border:1px solid {C['border_ui']};"
            f" border-radius:6px; }}")
        sh = QHBoxLayout()
        sh.addWidget(_lbl("😴  偷懶偵測", "white", 13, bold=True))
        sh.addStretch()
        self._slack_enable = QCheckBox("啟用")
        self._slack_enable.setChecked(bool(self._cfg.get("slack_enable", False)))
        self._slack_enable.toggled.connect(lambda c: self._cfg.update({"slack_enable": c}))
        sh.addWidget(self._slack_enable)
        slack_lay.addLayout(sh)

        scond = QHBoxLayout()
        scond.setSpacing(8)
        scond.addWidget(_lbl("連續", "gray", 12))
        self._slack_secs = QSpinBox()
        self._slack_secs.setRange(5, 3600)
        self._slack_secs.setValue(int(self._cfg.get("slack_secs", 60)))
        self._slack_secs.valueChanged.connect(lambda v: self._cfg.update({"slack_secs": v}))
        self._slack_secs.setFixedWidth(80)
        scond.addWidget(self._slack_secs)
        scond.addWidget(_lbl("秒　且　連續", "gray", 12))
        self._slack_count = QSpinBox()
        self._slack_count.setRange(2, 100)
        self._slack_count.setValue(int(self._cfg.get("slack_count", 3)))
        self._slack_count.valueChanged.connect(lambda v: self._cfg.update({"slack_count": v}))
        self._slack_count.setFixedWidth(70)
        scond.addWidget(self._slack_count)
        scond.addWidget(_lbl("筆 EXP 不變 → 警告", "gray", 12))
        scond.addStretch()
        slack_lay.addLayout(scond)

        hint = _lbl("（警告文字與觸發腳本在 ⚙ 設定中）", "gray", 10)
        slack_lay.addWidget(hint)
        self._slack_status = _lbl("未啟用", "gray", 12, mono=True)
        slack_lay.addWidget(self._slack_status)
        self._body_lay.addWidget(self._slack_card)

        # ─── Chart ────────────────────────────────────────────────────────
        ax_pen = pg.mkPen(C["chart_text"])

        # 圖表 1：EXP% 趨勢
        self._chart_pct_widget, cpct_lay = _card(12, 10)
        self._chart_pct_widget.setStyleSheet(
            f"QFrame {{ background:{C['bg_attr']}; border:1px solid {C['border_ui']};"
            f" border-radius:6px; }}")
        cpct_lay.addWidget(_lbl("EXP % 趨勢", "gray", 11))
        self._plot = pg.PlotWidget(background=C["plot_bg"], axisItems={"bottom": pg.DateAxisItem(orientation="bottom")})
        self._plot.setFixedHeight(180)
        self._plot.showGrid(x=True, y=True, alpha=0.12)
        self._plot.setLabel("left",   "EXP %",      color=C["chart_text"])
        self._plot.setLabel("bottom", "時間", color=C["chart_text"])
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
        self._chart_eps_widget.setStyleSheet(
            f"QFrame {{ background:{C['bg_attr']}; border:1px solid {C['border_ui']};"
            f" border-radius:6px; }}")
        ceps_lay.addWidget(_lbl("EXP / 秒  趨勢（每秒採樣近1分鐘均值）", "gray", 11))
        self._plot_eps = pg.PlotWidget(background=C["plot_bg"], axisItems={"bottom": pg.DateAxisItem(orientation="bottom")})
        self._plot_eps.setFixedHeight(160)
        self._plot_eps.showGrid(x=True, y=True, alpha=0.12)
        self._plot_eps.setLabel("left",   "EXP/s",    color=C["chart_text"])
        self._plot_eps.setLabel("bottom", "時間", color=C["chart_text"])
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
            f"QPushButton {{ background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            f"stop:0 #3aaa50,stop:1 #1a7a30); color:#ffffff;"
            f" border:1px solid #1a7a30; border-radius:5px; font-size:13px; font-weight:700; }}"
            f"QPushButton:hover {{ background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            f"stop:0 #4acc60,stop:1 #2a9a40); border-color:#2a9a40; }}")
        self._btn_start.clicked.connect(self._start)

        self._btn_stop = QPushButton("■  停止")
        self._btn_stop.setFixedHeight(38)
        self._btn_stop.setEnabled(False)
        self._btn_stop.setStyleSheet(
            f"QPushButton {{ background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            f"stop:0 #c03030,stop:1 #882020); color:#ffffff;"
            f" border:1px solid #882020; border-radius:5px; font-size:13px; font-weight:700; }}"
            f"QPushButton:hover {{ background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            f"stop:0 #e04040,stop:1 #b02828); border-color:#b02828; }}")
        self._btn_stop.clicked.connect(self._stop)

        btn_clear_log = QPushButton("清除紀錄")
        btn_clear_log.setFixedHeight(38)
        btn_clear_log.clicked.connect(self._clear_log)

        btn_clear_chart = QPushButton("清空圖表")
        btn_clear_chart.setFixedHeight(38)
        btn_clear_chart.clicked.connect(self._clear_charts)

        btn_diag = QPushButton("🔍 診斷擷取")
        btn_diag.setFixedHeight(38)
        btn_diag.setToolTip("擷取目前畫面與辨識結果，存到使用者資料夾供回報")
        btn_diag.clicked.connect(self._debug_capture)

        btn_calib = QPushButton("🎯 校準辨識")
        btn_calib.setFixedHeight(38)
        btn_calib.setToolTip("一次性校準：用你自己遊戲畫面建立辨識模板（換客戶端/字體時用）")
        btn_calib.clicked.connect(self._calibrate)

        ctrl_row.addWidget(self._btn_start)
        ctrl_row.addWidget(self._btn_stop)
        ctrl_row.addWidget(btn_clear_log)
        ctrl_row.addWidget(btn_clear_chart)
        ctrl_row.addWidget(btn_diag)
        ctrl_row.addWidget(btn_calib)
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
        self._slack_card.setVisible(self._vis.get("slack", True))
        self._chart_pct_widget.setVisible(self._vis.get("chart_pct", True))
        self._chart_eps_widget.setVisible(self._vis.get("chart_eps", True))
        self._log_widget.setVisible(self._vis.get("log", True))
        self.resize(sz)

    # ── 診斷擷取（給用 exe、沒有 Python 的人回報問題用）──────────────────────────
    def _debug_capture(self):
        import json, datetime, traceback
        if not _HAS_TEMPLATE:
            self._log_err("模板模組未載入，無法診斷擷取"); return
        try:
            out = os.path.join(_CFG_DIR, "EXPMonitor_debug")
            os.makedirs(out, exist_ok=True)
            hwnd, reg = find_window()
            if hwnd is None:
                self._log_warn("診斷擷取：找不到 MapleStory 視窗"); return
            img, cap = capture_strip(hwnd, reg)
            if img is None:
                self._log_err(f"診斷擷取：截圖失敗（{cap}）"); return
            ts = datetime.datetime.now().strftime("%H%M%S")
            _imwrite_u(os.path.join(out, f"{ts}_raw.png"), img)
            # 另存一張「未過濾的 mss 直擷底部條」，用來校準低 EXP% 定位
            mss_stats = {}
            try:
                import numpy as _np, cv2 as _cv2
                mss_img, _ = _mod._cap_mss(reg)
                if mss_img is not None:
                    _imwrite_u(os.path.join(out, f"{ts}_mss.png"), mss_img)
                    _hsv = _cv2.cvtColor(mss_img, _cv2.COLOR_BGR2HSV)
                    _yel = _cv2.inRange(_hsv, _mod.EXP_YELLOW_LO, _mod.EXP_YELLOW_HI)
                    mss_stats = {"mss_max": int(mss_img.max()),
                                 "mss_yellow_px": int((_yel > 0).sum())}
            except Exception as _e:
                mss_stats = {"mss_error": repr(_e)}
            y0, y1 = find_exp_bar_rows(img)
            band = img[y0:y1, :]
            _imwrite_u(os.path.join(out, f"{ts}_band.png"), band)
            ocr = (self._worker._ocr if (self._worker and getattr(self._worker, "_ocr", None))
                   else TemplateOCR())
            mask = build_mask(band)
            _imwrite_u(os.path.join(out, f"{ts}_mask.png"), mask)
            try:
                _imwrite_u(os.path.join(out, f"{ts}_block.png"), ocr._isolate_text(mask))
            except Exception:
                pass
            r = ocr.recognize(mask, debug=True)
            # 額外幾何資訊（判斷 DPI/座標是否錯位）
            geo = {}
            try:
                import win32gui as _wg
                wl, wt, wr, wb = _wg.GetWindowRect(hwnd)
                cr = _wg.GetClientRect(hwnd)
                cs = _wg.ClientToScreen(hwnd, (0, 0))
                geo = {"win_rect": [wl, wt, wr, wb],
                       "win_rect_wh": [wr - wl, wb - wt],
                       "client_wh": [cr[2] - cr[0], cr[3] - cr[1]],
                       "client_origin": [cs[0], cs[1]]}
            except Exception as _e:
                geo = {"geo_err": repr(_e)}
            try:
                import ctypes as _ct
                aw = _ct.c_int(0)
                _ct.windll.shcore.GetProcessDpiAwareness(0, _ct.byref(aw))
                geo["dpi_awareness"] = aw.value   # 0=unaware 1=system 2=permonitor
            except Exception:
                pass
            info = {
                "window": f"{reg['width']}x{reg['height']}",
                "geo": geo,
                "cap_method": cap,
                "strip_shape": list(img.shape),
                "exp_rows": f"{y0}-{y1}",
                "exp": r.get("exp"), "pct": r.get("pct"),
                "conf": round(r.get("conf", 0), 3), "reason": r.get("reason"),
                "n_runs": r.get("n_runs"), "widths": r.get("widths"),
                "templates_ready": ocr.is_ready(),
                "template_dir": str(ocr.template_dir),
                "mss": mss_stats,
            }
            with open(os.path.join(out, f"{ts}_info.txt"), "w", encoding="utf-8") as f:
                f.write(json.dumps(info, ensure_ascii=False, indent=2))
            self._log_info(
                f"診斷擷取完成 → {out}（exp={r.get('exp')} pct={r.get('pct')} "
                f"reason={r.get('reason')}）。請把整個資料夾傳回報。")
            try:
                os.startfile(out)   # 自動開啟資料夾（Windows）
            except Exception:
                pass
        except Exception as e:
            self._log_err(f"診斷擷取失敗：{e!r}")
            self._log_err(traceback.format_exc())

    def _calibrate(self):
        """一次性校準：抓現在的經驗列，請使用者輸入畫面上看到的數字，建立該客戶端字形的模板。"""
        if not _HAS_TEMPLATE:
            QMessageBox.warning(self, "校準", "辨識模組未載入，無法校準。"); return
        hwnd, reg = find_window()
        if hwnd is None:
            QMessageBox.warning(self, "校準", "找不到 MapleStory 視窗，請先開遊戲。"); return
        img, cap = capture_strip(hwnd, reg)
        if img is None:
            QMessageBox.warning(self, "校準", f"截圖失敗（{cap}）。"); return
        y0, y1 = find_exp_bar_rows(img)
        band = img[y0:y1, :]
        # 裁切到文字區塊讓預覽看得清楚
        try:
            mk = build_mask(band)
            cc = (mk > 0).sum(axis=0)
            import numpy as _np
            on = _np.where(cc > 16)[0]
            if len(on):
                bx0 = int(on[0] / 8) - 12
                bx1 = int(on[-1] / 8) + 12
                prev = band[:, max(0, bx0):min(band.shape[1], bx1)]
            else:
                prev = band
        except Exception:
            prev = band

        dlg = QDialog(self); dlg.setWindowTitle("校準辨識（一次性）")
        dlg.setMaximumWidth(860)
        v = QVBoxLayout(dlg)
        v.addWidget(_lbl("下圖是目前擷取到的『經驗列』。請照畫面上看到的、輸入 EXP 數字與百分比：", "white", 12))
        rgb = cv2.cvtColor(prev, cv2.COLOR_BGR2RGB)
        hh, ww = rgb.shape[:2]
        qimg = QImage(rgb.tobytes(), ww, hh, 3 * ww, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        # 經驗列又寬又矮：寬度上限 800，高度拉到約 56px 方便辨識（不保持比例沒關係，預覽用）
        disp_w = min(800, ww)
        lblimg = QLabel()
        lblimg.setPixmap(pix.scaled(disp_w, 56))
        lblimg.setStyleSheet("background:#000;")
        v.addWidget(lblimg)
        exp_in = QLineEdit(); exp_in.setPlaceholderText("EXP 數字（不含逗號），例：35140164989579")
        pct_in = QLineEdit(); pct_in.setPlaceholderText("百分比，例：86.311")
        v.addWidget(_lbl("EXP 數字（不含逗號）", "gray", 11)); v.addWidget(exp_in)
        v.addWidget(_lbl("百分比", "gray", 11)); v.addWidget(pct_in)
        v.addWidget(_lbl("提示：一個數字通常涵蓋不到 0~9 全部，依提示對「含缺少數字」的畫面多校準幾張即可。", "gray", 10))
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        v.addWidget(bb)
        if dlg.exec_() != QDialog.Accepted:
            return
        try:
            ok, msg = build_user_templates(band, exp_in.text(), pct_in.text())
        except Exception as e:
            QMessageBox.critical(self, "校準失敗", repr(e)); return
        (QMessageBox.information if ok else QMessageBox.warning)(self, "校準結果", msg)
        self._log_info("校準：" + msg)
        # 立即套用：重載辨識器，讓 worker 下一幀就用新模板
        try:
            if self._worker is not None and getattr(self._worker, "_ocr", None) is not None:
                self._worker._ocr = TemplateOCR()   # 會優先載入使用者模板
        except Exception:
            pass

    # ── 偷懶偵測 ────────────────────────────────────────────────────────────────
    def _render_alerts(self):
        self._slack_banner.setText(self._slack_alert or "")
        self._slack_banner.setVisible(bool(self._slack_alert))
        self._loweff_banner.setText(self._loweff_alert or "")
        self._loweff_banner.setVisible(bool(self._loweff_alert))

    def _check_loweff(self):
        thr = 0.0
        try:
            thr = float(self._cfg.get("loweff_threshold", 0) or 0)
        except (TypeError, ValueError):
            thr = 0.0
        if not self._cfg.get("loweff_enable", False) or thr <= 0:
            if self._loweff_alert is not None:
                self._loweff_alert = None
                self._render_alerts()
            return
        eps = self._rate.exp_per_sec
        if eps is None or self._rate.sample_count < 8:
            return   # 樣本不足，暫不判斷
        if eps < thr:
            self._loweff_alert = f"🐢 效率過低：{eps:,.0f} EXP/s（門檻 {thr:,.0f}）"
        else:
            self._loweff_alert = None
        self._render_alerts()

    def _slack_normal_style(self):
        self._slack_card.setStyleSheet(
            f"QFrame {{ background:{C['bg2']}; border:1px solid {C['border']};"
            f" border-radius:10px; }}")

    def _set_slack_status(self, text, color):
        self._slack_status.setText(text)
        self._slack_status.setStyleSheet(
            f"color:{color}; font-size:12px; font-family:Consolas;"
            f" background:transparent; border:none;")

    def _check_slack(self, exp_int, ts):
        if not self._slack_enable.isChecked():
            self._slack_normal_style()
            self._set_slack_status("未啟用", C["gray"])
            if self._slack_alert is not None:
                self._slack_alert = None
                self._render_alerts()
            return
        now = time.time()
        # EXP 有變 → 正常成長，歸零
        if self._slack_last_exp is None or exp_int != self._slack_last_exp:
            self._slack_last_exp  = exp_int
            self._slack_t0        = now
            self._slack_streak    = 1
            self._slack_triggered = False
            self._slack_normal_style()
            self._set_slack_status("✓ 經驗正常成長", C["green"])
            self._slack_alert = None
            self._render_alerts()
            return
        # EXP 不變 → 累計
        self._slack_streak += 1
        elapsed = now - (self._slack_t0 or now)
        need_n  = self._slack_count.value()
        need_s  = self._slack_secs.value()
        if self._slack_streak >= need_n and elapsed >= need_s:
            if not self._slack_triggered:
                self._slack_triggered = True
                self._fire_slack_warning(ts, elapsed)
        else:
            self._set_slack_status(
                f"EXP 未變：{self._slack_streak} 筆 / {elapsed:.0f}s"
                f"  (門檻 {need_n} 筆 且 {need_s}s)", C["yellow"])

    def _fire_slack_warning(self, ts, elapsed):
        self._slack_card.setStyleSheet(
            f"QFrame {{ background:#3a1414; border:2px solid {C['red']};"
            f" border-radius:10px; }}")
        self._set_slack_status(f"⚠ 偷懶警告！EXP 已停滯 {elapsed:.0f}s", C["red"])
        self._log_colored(
            f"[{ts}]  😴 ！偷懶警告！ EXP 已 {elapsed:.0f}s 沒有增加", C["red"])
        self._run_slack_actions(elapsed)

    def _run_slack_actions(self, elapsed):
        """跳出自訂文字視窗 + 執行使用者指定的觸發腳本（非阻塞）。"""
        exp = self._slack_last_exp
        pct = self._prev_pct
        # 自訂警告文字（支援 {sec} {exp} {pct}；以及字面的反斜線n換行）
        raw = (self._cfg.get("slack_msg") or "").strip() or "！偷懶警告！EXP 已 {sec} 秒沒有增加"
        try:
            msg = raw.format(sec=int(elapsed),
                             exp=(f"{exp:,}" if exp else "-"),
                             pct=(f"{pct:.3f}" if pct else "-"))
        except Exception:
            msg = raw
        msg = msg.replace(chr(92) + "n", chr(10))   # 讓使用者可用 \n 換行
        try:
            QApplication.alert(self, 3000)           # 閃爍工作列（不影響 OBS）
        except Exception:
            pass
        # 在 UI 上顯示橫幅（OBS 視窗擷取抓得到），不再用彈窗
        self._slack_alert = msg.replace(chr(10), "　")
        self._render_alerts()
        # 使用者觸發腳本：情境用環境變數傳入，shell=True 讓使用者自由填指令
        cmd = (self._cfg.get("slack_cmd") or "").strip()
        if cmd:
            try:
                env = os.environ.copy()
                env["SLACK_SECONDS"] = str(int(elapsed))
                env["SLACK_EXP"] = str(exp if exp is not None else "")
                env["SLACK_PCT"] = (f"{pct:.3f}" if pct is not None else "")
                subprocess.Popen(cmd, shell=True, env=env)
                self._log_colored(f"   ↳ 已執行觸發腳本：{cmd}", C["gray"])
            except Exception as e:
                self._log_colored(f"   ↳ 觸發腳本失敗：{e}", C["red"])

    def _slack_test(self):
        self._log_colored("[測試]  手動觸發偷懶動作（不影響實際偵測狀態）", C["yellow"])
        self._run_slack_actions(self._slack_secs.value())

    # ── Control ───────────────────────────────────────────────────────────────
    def _start(self):
        if self._thread and self._thread.isRunning(): return
        self._save_config()
        self._prev_pct        = None
        self._prev_exp_int    = None
        self._slack_last_exp  = None
        self._slack_streak    = 0
        self._slack_t0        = None
        self._slack_triggered = False
        self._exp_fail_streak = 0
        self._max_exp_est     = None
        self._sess_start_ts   = time.time()
        self._sess_start_pct  = None
        self._sess_prev_pct   = None
        self._sess_levels     = 0
        self._sess_start_lbl.setText("起始 —")
        self._sess_dur_lbl.setText("持續 00:00:00")
        self._sess_gain_lbl.setText("增加 +0.000%")
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
        self._worker.error_sig.connect(
            lambda m: self._set_status("⚠ 錯誤（見下方紀錄）", C["red"]))
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
        # EXP 直接採用 OCR（模板比對）讀到的完整數字。
        # 模板 OCR 已可靠讀出完整 15 位數；異常值交給下面的
        # max_exp 一致性 / EXP 遞減 / 位數守衛擋掉（誤報不會畫進圖）。
        exp_int = None
        if exp:
            try:
                _v = int(exp.replace(",", ""))
                if _v > 1_000_000_000:
                    exp_int = _v
            except Exception:
                pass

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
                f"color:{C['white_hero']}; font-size:44px; font-weight:700;"
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
                self._check_slack(exp_int, ts)

            diff_str = ""
            if self._prev_pct is not None:
                diff_str = f"  ({pct_f - self._prev_pct:+.3f}%)"

            line = f"[{ts}]  {pct}%"
            if exp:     line += f"  {exp}"
            if diff_str: line += diff_str
            line += f"  [{cap}]"
            self._log_colored(line, C["green"])

            self._update_session(pct_f)
            self._prev_pct = pct_f
            if exp_int is not None:
                self._prev_exp_int = exp_int
        else:
            self._pct_lbl.setStyleSheet(
                f"color:{C['accent']}; font-size:44px; font-weight:700;"
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
                C["cyan"] if pph >= 0 else C["red"])
        self._ttl_widget.set_value(ttl, C["cyan"])
        self._check_loweff()

    # ── Chart ─────────────────────────────────────────────────────────────────
    def _clear_charts(self):
        self._curve.setData([], [])
        self._curve_eps.setData([], [])
        self._rate.clear_chart_data()

    def _update_session(self, pct_f):
        # 起始
        if self._sess_start_pct is None:
            self._sess_start_pct = pct_f
            self._sess_prev_pct = pct_f
            self._sess_levels = 0
            self._sess_start_lbl.setText(f"起始 {pct_f:.3f}%")
            self._sess_gain_lbl.setText("增加 +0.000%")
            return
        prev = self._sess_prev_pct
        if prev > 90.0 and pct_f < 10.0:
            # 真升等（% 由接近滿掉到接近 0）
            self._sess_levels += 1
        elif abs(pct_f - prev) > 10.0:
            # 不合理的單幀跳動 → 視為 OCR 誤判，忽略（不更新基準、不刷新）
            return
        self._sess_prev_pct = pct_f
        gain = self._sess_levels * 100.0 + (pct_f - self._sess_start_pct)
        if gain < 0:
            gain = 0.0
        self._sess_gain_lbl.setText(f"增加 +{gain:.3f}%")

    def _refresh_chart(self):
        if self._sess_start_ts is not None and self._btn_stop.isEnabled():
            el = int(time.time() - self._sess_start_ts)
            self._sess_dur_lbl.setText(
                f"持續 {el // 3600:02d}:{(el % 3600) // 60:02d}:{el % 60:02d}")
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
        self._log_colored(f"i  {msg}", C["white"])

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


    # ── 主題切換 ─────────────────────────────────────────────────────────────
    def _apply_theme(self, name: str):
        """切換 UI 主題：name = 'dark' | 'maplestory'"""
        global C
        if name not in THEMES:
            return
        C.update(THEMES[name])
        self._cfg["theme"] = name

        # 重建並套用 QSS
        self.setStyleSheet(_build_qss())

        # 更新 pyqtgraph 全域前景色
        pg.setConfigOptions(foreground=C["chart_text"])

        # Header 漸層
        self._hdr.setStyleSheet(
            f"QFrame {{ background: qlineargradient(x1:0, y1:0, x2:0, y2:1,"
            f" stop:0 {C['hdr_top']}, stop:1 {C['hdr_bot']});"
            f" border-bottom: 1px solid {C['border']}; }}")

        # 標題標籤
        self._title_lb.setStyleSheet(
            f"color:{C['white']}; font-size:15px; font-weight:700;"
            f" font-family:'微軟正黑體'; background:transparent; border:none;")

        # Body 背景
        self._body_widget.setStyleSheet(f"background:{C['bg']};")

        # EXP hero 卡
        self._exp_card.setStyleSheet(
            f"QFrame {{ background:{C['bg_hero']}; border:1px solid {C['border_ui']};"
            f" border-radius:6px; }}")

        # StatTiles
        for tile in [self._tile_eps, self._tile_pph, self._ttl_widget]:
            tile.refresh_theme()

        # 偷懶 card + 圖表 cards（bg_attr）
        _attr_ss = (f"QFrame {{ background:{C['bg_attr']}; border:1px solid {C['border_ui']};"
                    f" border-radius:6px; }}")
        self._slack_card.setStyleSheet(_attr_ss)
        self._chart_pct_widget.setStyleSheet(_attr_ss)
        self._chart_eps_widget.setStyleSheet(_attr_ss)

        # 設定面板框
        self._settings.setStyleSheet(
            f"QFrame {{ background:{C['bg_attr']}; border:1px solid {C['border_ui']};"
            f" border-radius:6px; }}")

        # 圖表軸文字、背景
        ax_pen = pg.mkPen(C["chart_text"])
        for plot_w in [self._plot, self._plot_eps]:
            plot_w.setBackground(C["plot_bg"])
            plot_w.getAxis("left").setTextPen(ax_pen)
            plot_w.getAxis("bottom").setTextPen(ax_pen)
            plot_w.getAxis("left").setPen(ax_pen)
            plot_w.getAxis("bottom").setPen(ax_pen)

        # 軸標題（setLabel 用獨立 color 參數）
        self._plot.setLabel("left",   "EXP %",   color=C["chart_text"])
        self._plot.setLabel("bottom", "時間",     color=C["chart_text"])
        self._plot_eps.setLabel("left",   "EXP/s", color=C["chart_text"])
        self._plot_eps.setLabel("bottom", "時間",  color=C["chart_text"])

        # 曲線顏色
        self._curve.setPen(pg.mkPen(C["cyan"], width=2))
        self._curve.setSymbolBrush(pg.mkBrush(C["cyan"]))
        self._curve_eps.setPen(pg.mkPen(C["green"], width=2))
        self._curve_eps.setSymbolBrush(pg.mkBrush(C["green"]))

        # EXP % 大字
        self._pct_lbl.setStyleSheet(
            f"color:{C['white_hero']}; font-size:44px; font-weight:700;"
            f" font-family:Consolas; background:transparent;")

        # 狀態列文字（inline styled，非 _lbl）
        self._status_lbl.setStyleSheet(
            f"color:{C['gray']}; font-size:12px; background:transparent; border:none;")
        self._status_dot.setStyleSheet(
            f"color:{C['gray']}; font-size:15px; background:transparent; border:none;")

        # 重新 polish 所有 QLabel（刷新 lbl_clr property 顏色）
        from PyQt5.QtWidgets import QLabel as _QLabel
        for lb in self.findChildren(_QLabel):
            lb.style().unpolish(lb)
            lb.style().polish(lb)
            lb.update()

    def _load_config(self):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data.get("cfg"), dict):
                self._cfg.update(data["cfg"])
            if isinstance(data.get("vis"), dict):
                self._vis.update(data["vis"])
        except Exception:
            pass

    def _save_config(self):
        try:
            os.makedirs(_CFG_DIR, exist_ok=True)
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump({"cfg": self._cfg, "vis": self._vis}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def closeEvent(self, event):
        self._save_config()
        self._stop()
        event.accept()


if __name__ == "__main__":
    # ── DPI 感知必須在 QApplication 之前設定 ──────────────────────────────
    # 否則 GetClientRect/ClientToScreen 會回傳「邏輯像素」(被縮放虛擬化),
    # 與 mss 抓的「物理像素」對不上 → 算出的視窗底部偏高 → EXP 條被切掉。
    try:
        set_dpi_awareness()
    except Exception:
        pass
    try:
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    except Exception:
        pass
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
