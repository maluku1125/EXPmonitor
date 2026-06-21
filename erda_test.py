#!/usr/bin/env python3
"""
erda_test.py — Erda（氣息/碎片/角標）面板辨識「獨立測試程式」
擷取截圖 → 可放大 → 逐欄框選(氣息/碎片/角標) → 校準 erda 字體 → 讀數。
需求：pip install PyQt5 numpy opencv-python mss pywin32
"""
import os, sys, importlib.util
import numpy as np
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton,
                             QVBoxLayout, QHBoxLayout, QGridLayout, QFrame, QScrollArea, QLineEdit)
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor

_HERE = os.path.dirname(os.path.abspath(__file__))
def _load(name):
    sp = importlib.util.spec_from_file_location(name, os.path.join(_HERE, name + ".py"))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
M = _load("exp_monitor")
T = _load("exp_template_ocr")
ERDA_TMPL_DIR = os.path.join(_HERE, "templates_erda")
FIELDS = ["氣息", "碎片", "角標"]
FCOLOR = {"氣息": "#3aa0ff", "碎片": "#ffb020", "角標": "#ff5070"}


def grab_client():
    hwnd, reg = M.find_window()
    if hwnd is None:
        return None
    try:
        import mss
        with mss.mss() as sct:
            raw = sct.grab({"left": reg["left"], "top": reg["top"],
                            "width": reg["width"], "height": reg["height"]})
            return cv2.cvtColor(np.array(raw, dtype=np.uint8), cv2.COLOR_BGRA2BGR)
    except Exception as e:
        print("grab error:", e); return None


# 各欄位濾鏡門檻 (gray>, sat<)：氣息是白字疊亮藍條→要嚴(排藍頭)；碎片是淺紫底深→寬鬆
ERDA_MASK = {"氣息": (165, 70), "碎片": (150, 120), "角標": (150, 120)}

def erda_mask(crop, field="碎片"):
    gth, sth = ERDA_MASK.get(field, (150, 120))
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV); gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return T._upscale(((gray > gth) & (hsv[:, :, 1] < sth)).astype(np.uint8) * 255)


def _segments(mask):
    cs = np.sum(mask > 0, axis=0); on = np.where(cs > 0)[0]
    if len(on) == 0:
        return mask, []
    mask = mask[:, max(0, on[0] - 4):on[-1] + 5]
    runs = [list(r) for r in T.TemplateOCR._runs(mask)]
    hs = []
    for a, b in runs:
        rows = np.where(np.any(mask[:, a:b + 1] > 0, axis=1))[0]
        hs.append(rows[-1] - rows[0] + 1 if len(rows) else 0)
    if hs:
        med = np.median([h for h in hs if h > 0]) or 1
        runs = [r for r, h in zip(runs, hs) if h >= 0.55 * med and (r[1] - r[0] + 1) >= 4]
    return mask, runs


class ErdaOCR:
    CHARSET = list("0123456789/")
    def __init__(self, d): self.dir = d; self.tmpl = {}; self.load()
    def _fn(self, ch): return "slash" if ch == "/" else ch
    def load(self):
        self.tmpl = {}
        for ch in self.CHARSET:
            p = os.path.join(self.dir, self._fn(ch) + ".png")
            if os.path.exists(p):
                im = T._imread_u(p, cv2.IMREAD_GRAYSCALE)
                if im is not None: self.tmpl[ch] = im
    def ready(self): return sum(c.isdigit() for c in self.tmpl) >= 10
    def read(self, crop, field):
        mask, runs = _segments(erda_mask(crop, field))
        if not self.tmpl: return ""
        out = ""
        for a, b in runs:
            seg = T.TemplateOCR._crop(mask[:, a:b + 1]); sf = seg.astype(np.float32) / 255.0
            best, bs = None, -2.0
            for ch, t in self.tmpl.items():
                tc = T.TemplateOCR._crop(t)
                rz = cv2.resize(sf, (tc.shape[1], tc.shape[0]), interpolation=cv2.INTER_AREA)
                aa = rz.flatten(); cc = (tc.astype(np.float32) / 255.0).flatten()
                if aa.std() < 0.01 or cc.std() < 0.01: continue
                v = float(np.corrcoef(aa, cc)[0, 1])
                if v > bs: bs, best = v, ch
            if best: out += best
        return out
    def calibrate(self, crop, truth, samples, field):
        mask, runs = _segments(erda_mask(crop, field))
        for cand in (truth, truth.replace("/", "")):
            if len(runs) == len(cand):
                for (a, b), ch in zip(runs, cand):
                    if ch in self.CHARSET:
                        samples.setdefault(ch, []).append(T.TemplateOCR._crop(mask[:, a:b + 1]))
                return True, len(runs)
        return False, len(runs)
    def save(self, samples):
        os.makedirs(self.dir, exist_ok=True)
        for ch, segs in samples.items():
            H = int(np.median([s.shape[0] for s in segs])); W = int(np.median([s.shape[1] for s in segs]))
            rez = [cv2.resize(s, (W, H), interpolation=cv2.INTER_AREA) for s in segs]
            T._imwrite_u(os.path.join(self.dir, self._fn(ch) + ".png"), np.mean(rez, 0).astype(np.uint8))
        self.load()


BADGE_DIR = os.path.join(_HERE, "templates_erda_badge")

def badge_mask(crop):
    """角標：取白色數字（亮、低飽和），裁到內容、統一大小，供整體比對。"""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV); gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mk = ((gray > 175) & (hsv[:, :, 1] < 95)).astype(np.uint8) * 255
    bw = mk > 0; rows = np.any(bw, 1); cols = np.any(bw, 0)
    if rows.any() and cols.any():
        r0, r1 = np.where(rows)[0][[0, -1]]; c0, c1 = np.where(cols)[0][[0, -1]]
        mk = mk[r0:r1 + 1, c0:c1 + 1]
    return cv2.resize(mk, (64, 40), interpolation=cv2.INTER_AREA) if mk.size else np.zeros((40, 64), np.uint8)


class BadgeMatcher:
    """角標 0~20：整張圖比對 21 個範本，取最相似者。"""
    def __init__(self, d): self.dir = d; self.refs = {}; self.load()
    def load(self):
        self.refs = {}
        if not os.path.isdir(self.dir): return
        for fn in os.listdir(self.dir):
            if fn.endswith(".png"):
                try: val = int(os.path.splitext(fn)[0])
                except ValueError: continue
                im = T._imread_u(os.path.join(self.dir, fn), cv2.IMREAD_GRAYSCALE)
                if im is not None: self.refs[val] = cv2.resize(im, (64, 40))
    def count(self): return len(self.refs)
    def save(self, val, crop):
        """保留原圖到 raw/，再從該值所有原圖重算平均遮罩當範本（之後可重調門檻不必重抓）。"""
        rawdir = os.path.join(self.dir, "raw")
        os.makedirs(rawdir, exist_ok=True)
        i = 1
        while os.path.exists(os.path.join(rawdir, f"{val}_{i:02d}.png")):
            i += 1
        T._imwrite_u(os.path.join(rawdir, f"{val}_{i:02d}.png"), crop)
        masks = []
        for fn in os.listdir(rawdir):
            if fn.startswith(f"{val}_") and fn.endswith(".png"):
                im = T._imread_u(os.path.join(rawdir, fn), cv2.IMREAD_COLOR)
                if im is not None:
                    masks.append(badge_mask(im).astype(np.float32))
        if masks:
            T._imwrite_u(os.path.join(self.dir, f"{val}.png"),
                         np.mean(masks, 0).astype(np.uint8))
        self.load()
        return i   # 該值目前累積張數

    def compose_from_digits(self, erda_ocr):
        """用已校準的 0~9 數字模板，拼出 0~20 全部角標範本（免手動收集到 20）。"""
        digs = {c: erda_ocr.tmpl[c] for c in "0123456789" if c in erda_ocr.tmpl}
        miss = [c for c in "0123456789" if c not in digs]
        if miss:
            return False, miss
        os.makedirs(self.dir, exist_ok=True)
        for V in range(21):
            glyphs = [T.TemplateOCR._crop(digs[c]) for c in str(V)]
            Hc = max(g.shape[0] for g in glyphs)
            gap = max(2, int(Hc * 0.12))
            rez = []
            for g in glyphs:
                w = max(1, int(g.shape[1] * Hc / g.shape[0]))
                rez.append(cv2.resize(g, (w, Hc), interpolation=cv2.INTER_AREA))
            tw = sum(r.shape[1] for r in rez) + gap * (len(rez) - 1)
            canvas = np.zeros((Hc, tw), np.uint8); x = 0
            for r in rez:
                canvas[:, x:x + r.shape[1]] = r; x += r.shape[1] + gap
            T._imwrite_u(os.path.join(self.dir, f"{V}.png"),
                         cv2.resize(canvas, (64, 40), interpolation=cv2.INTER_AREA))
        self.load()
        return True, []
    def read(self, crop):
        if not self.refs: return None, 0.0
        n = badge_mask(crop).astype(np.float32) / 255.0
        best, bs = None, -2.0
        for val, ref in self.refs.items():
            r = ref.astype(np.float32) / 255.0
            if n.std() < 0.01 or r.std() < 0.01: continue
            sc = float(np.corrcoef(n.flatten(), r.flatten())[0, 1])
            if sc > bs: bs, best = sc, val
        return best, bs


def to_pix(bgr, scale):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); h, w = rgb.shape[:2]
    pm = QPixmap.fromImage(QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format_RGB888))
    return pm.scaled(max(1, int(w * scale)), max(1, int(h * scale)))


class ShotLabel(QLabel):
    """顯示截圖(可縮放)；拖框選取，座標換算回原圖。"""
    def __init__(self, on_select):
        super().__init__()
        self._on_select = on_select; self._scale = 1.0
        self._boxes = {}; self._start = None; self._cur = None
        self.setStyleSheet("background:#000;"); self.setAlignment(Qt.AlignTop | Qt.AlignLeft)
    def render(self, bgr, scale, boxes):
        self._scale = scale; self._boxes = boxes
        self.setPixmap(to_pix(bgr, scale)); self.setFixedSize(self.pixmap().size())
    def mousePressEvent(self, e):
        if self.pixmap() is None: return
        self._start = e.pos(); self._cur = e.pos(); self.update()
    def mouseMoveEvent(self, e):
        if self._start is not None: self._cur = e.pos(); self.update()
    def mouseReleaseEvent(self, e):
        if self._start is None: return
        r = QRect(self._start, e.pos()).normalized(); self._start = self._cur = None
        s = self._scale
        orig = (int(r.x()/s), int(r.y()/s), int(r.width()/s), int(r.height()/s))
        if orig[2] > 3 and orig[3] > 3: self._on_select(orig)
        self.update()
    def paintEvent(self, ev):
        super().paintEvent(ev)
        if self.pixmap() is None: return
        p = QPainter(self); s = self._scale
        for f, (x, y, w, h) in self._boxes.items():
            p.setPen(QPen(QColor(FCOLOR[f]), 2))
            p.drawRect(int(x*s), int(y*s), int(w*s), int(h*s))
            p.drawText(int(x*s), int(y*s) - 3, f)
        if self._start is not None and self._cur is not None:
            p.setPen(QPen(QColor("#ffff00"), 1, Qt.DashLine))
            p.drawRect(QRect(self._start, self._cur).normalized())
        p.end()


class ErdaTest(QMainWindow):
    def __init__(self):
        super().__init__()
        M.set_dpi_awareness()
        self.setWindowTitle("Erda 面板辨識 — 測試")
        self.resize(1040, 800)
        self._erda = ErdaOCR(ERDA_TMPL_DIR)
        self._badge = BadgeMatcher(BADGE_DIR)
        self._shot = None; self._fit = 1.0; self._zoom = 1.0
        self._regions = {}; self._sel = None
        root = QWidget(); self.setCentralWidget(root); lay = QVBoxLayout(root)

        top = QHBoxLayout()
        for txt, fn in [("① 擷取截圖", self._shoot), ("＋放大", lambda: self._set_zoom(1.5)),
                        ("－縮小", lambda: self._set_zoom(1/1.5)), ("100%", lambda: self._set_zoom(0))]:
            b = QPushButton(txt); b.clicked.connect(fn); top.addWidget(b)
        for f in FIELDS:
            b = QPushButton(f"框選 {f}"); b.clicked.connect(lambda _, x=f: self._begin(x)); top.addWidget(b)
        b = QPushButton("② 讀取"); b.clicked.connect(self._read); top.addWidget(b)
        bd = QPushButton("存圖"); bd.setToolTip("把框好的三塊原圖+遮罩存到 erda_debug 供回報"); bd.clicked.connect(self._dump); top.addWidget(bd)
        top.addStretch(); lay.addLayout(top)

        cal = QHBoxLayout()
        cal.addWidget(QLabel("校準 — 氣息(如896/1000):"))
        self._in_b = QLineEdit(); self._in_b.setFixedWidth(110); cal.addWidget(self._in_b)
        cal.addWidget(QLabel("碎片(如4547):"))
        self._in_s = QLineEdit(); self._in_s.setFixedWidth(110); cal.addWidget(self._in_s)
        bc = QPushButton("校準(建字體)"); bc.clicked.connect(self._calibrate); cal.addWidget(bc)
        cal.addSpacing(16); cal.addWidget(QLabel("角標值(0~20):"))
        self._in_badge = QLineEdit(); self._in_badge.setFixedWidth(50); cal.addWidget(self._in_badge)
        bbs = QPushButton("存角標樣本"); bbs.clicked.connect(self._save_badge); cal.addWidget(bbs)
        bcm = QPushButton("用數字合成0~20"); bcm.setToolTip("用已校準的0~9拼出全部角標範本，免手動收集"); bcm.clicked.connect(self._compose_badge); cal.addWidget(bcm)
        cal.addStretch(); lay.addLayout(cal)

        self._hint = QLabel("① 擷取 → ＋放大到看得清 → 「框選 氣息/碎片/角標」各拖一個緊框 → 校準 → ② 讀取")
        self._hint.setStyleSheet("color:#888;"); lay.addWidget(self._hint)

        self._lbl = ShotLabel(self._on_select)
        sca = QScrollArea(); sca.setWidget(self._lbl); sca.setWidgetResizable(False)
        lay.addWidget(sca, 1)

        res = QFrame(); res.setStyleSheet("background:#161b22;border:1px solid #30363d;border-radius:6px;")
        rg = QGridLayout(res); self._res = {}; self._mlbl = {}
        for i, f in enumerate(FIELDS):
            t = QLabel(f); t.setStyleSheet("color:#aaa;"); rg.addWidget(t, i, 0)
            v = QLabel("—"); v.setStyleSheet("color:#e6edf3;font-size:18px;font-family:Consolas;")
            self._res[f] = v; rg.addWidget(v, i, 1)
            mlb = QLabel(); self._mlbl[f] = mlb; rg.addWidget(mlb, i, 2)
        lay.addWidget(res)
        self._title()

    def _title(self):
        self.setWindowTitle("Erda 測試 " + ("（已校準）" if self._erda.ready() else "（未校準，讀取會不準）"))
    def _scale(self): return self._fit * self._zoom
    def _rerender(self):
        if self._shot is not None:
            self._lbl.render(self._shot, self._scale(), dict(self._regions))
    def _set_zoom(self, factor):
        if self._shot is None: return
        self._zoom = 1.0 if factor == 0 else max(0.2, min(8.0, self._zoom * factor))
        self._rerender()
        self._hint.setText(f"目前縮放 {self._scale():.2f}x")
    def _shoot(self):
        bgr = grab_client()
        if bgr is None: self._hint.setText("找不到 MapleStory 視窗"); return
        self._shot = bgr; self._fit = min(1.0, 980 / bgr.shape[1]); self._zoom = 1.0
        self._rerender()
        self._hint.setText(f"已擷取 {bgr.shape[1]}x{bgr.shape[0]}。可＋放大再框選。")
    def _begin(self, f):
        if self._shot is None: self._hint.setText("請先擷取截圖"); return
        self._sel = f; self._hint.setText(f"在截圖上拖一個「緊貼數字」的框（{f}）")
    def _on_select(self, box):
        if self._sel is None: self._hint.setText("請先按「框選 …」"); return
        self._regions[self._sel] = box; f = self._sel; self._sel = None
        self._rerender(); self._hint.setText(f"已框 {f}={box}。可框其他欄位 / 校準 / 讀取")
    def _calibrate(self):
        if self._shot is None: self._hint.setText("請先擷取"); return
        samples = {}; msgs = []
        for f, key in [("氣息", self._in_b), ("碎片", self._in_s)]:
            truth = key.text().strip()
            if not truth: continue
            if f not in self._regions: msgs.append(f"{f}:未框"); continue
            x, y, w, h = self._regions[f]
            ok, n = self._erda.calibrate(self._shot[y:y+h, x:x+w], truth, samples, f)
            msgs.append(f"{f}:{'OK' if ok else f'切{n}段≠{len(truth)}字(框緊一點)'}")
        if not samples: self._hint.setText("請框好欄位並輸入正確值"); return
        self._erda.save(samples); self._title()
        have = [c for c in "0123456789" if c in self._erda.tmpl]
        miss = [c for c in "0123456789" if c not in have]
        self._hint.setText("校準 " + " / ".join(msgs) + f"；數字 {len(have)}/10" +
                           (f"，缺 {','.join(miss)}" if miss else "，齊全！"))
    def _dump(self):
        if self._shot is None or not self._regions:
            self._hint.setText("請先擷取並框選至少一欄"); return
        out = os.path.join(_HERE, "erda_debug"); os.makedirs(out, exist_ok=True)
        for f, (x, y, w, h) in self._regions.items():
            crop = self._shot[y:y+h, x:x+w]
            T._imwrite_u(os.path.join(out, f"{f}_raw.png"), crop)
            T._imwrite_u(os.path.join(out, f"{f}_mask.png"), erda_mask(crop, f))
        self._hint.setText(f"已存到 {out}（{ '、'.join(self._regions) }）。把整個 erda_debug 資料夾給開發者。")

    def _compose_badge(self):
        ok, miss = self._badge.compose_from_digits(self._erda)
        if ok:
            self._hint.setText("已用 0~9 合成角標範本 0~20（共 21 個）。直接「② 讀取」即可。")
        else:
            self._hint.setText(f"數字模板還缺 {','.join(miss)} → 請先把氣息/碎片校準到 0~9 齊全")

    def _save_badge(self):
        if self._shot is None or "角標" not in self._regions:
            self._hint.setText("請先擷取並框選『角標』"); return
        v = self._in_badge.text().strip()
        if not v.isdigit() or not (0 <= int(v) <= 20):
            self._hint.setText("請在『角標值』輸入 0~20 的整數"); return
        x, y, w, h = self._regions["角標"]
        n = self._badge.save(int(v), self._shot[y:y+h, x:x+w])
        have = sorted(self._badge.refs)
        miss = [str(i) for i in range(21) if i not in self._badge.refs]
        self._hint.setText(f"已存角標 {v}（第{n}張）。目前 {len(have)}/21，缺：{','.join(miss) or '無，已齊全！'}")

    def _read(self):
        if self._shot is None: self._hint.setText("請先擷取"); return
        for f in FIELDS:
            if f not in self._regions: self._res[f].setText("（未框）"); continue
            x, y, w, h = self._regions[f]; crop = self._shot[y:y+h, x:x+w]
            raw = self._erda.read(crop, f)   # 有學到的字就先讀（不必等 10/10）
            if f == "氣息":
                d = raw.split("/")[0] if "/" in raw else raw
                if "/" not in raw and d.endswith("1000") and len(d) > 4: d = d[:-4]
                raw = (d + " / 1000") if d else "—"
            elif f == "角標":
                bval, bsc = self._badge.read(crop)
                raw = (f"{bval}  (比對 {bsc:.2f})" if bval is not None else "（角標範本未建）")
            self._res[f].setText(raw or "—")
            mask, _ = _segments(erda_mask(crop, f))
            if mask.size:
                mm = cv2.resize(mask, (min(360, max(40, mask.shape[1])), 40), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(mm, cv2.COLOR_GRAY2RGB)
                self._mlbl[f].setPixmap(QPixmap.fromImage(
                    QImage(rgb.tobytes(), rgb.shape[1], rgb.shape[0], 3*rgb.shape[1], QImage.Format_RGB888)))
        self._hint.setText("讀取完成。" + ("" if self._erda.ready() else "（尚未校準→先校準）"))


if __name__ == "__main__":
    app = QApplication(sys.argv); app.setStyle("Fusion")
    w = ErdaTest(); w.show(); sys.exit(app.exec_())
