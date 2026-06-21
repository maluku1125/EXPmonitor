"""
erda_ocr.py — Erda 面板（氣息 / 碎片 / 氣息數量）辨識核心
=====================================================
- 氣息(白字疊亮藍條)、碎片(淺紫底深) 各用不同濾鏡 → 白字遮罩 → 模板比對讀數。
- 氣息數量(0~20，7px 太小) → 整張圖跟 21 個範本「整體比對」取最像；範本可由 0~9 數字合成。
- ErdaReader：吃整個 client 畫面 + 已存區域 → 回傳 {breath, shred, badge}，含時間平滑。
依賴 exp_template_ocr（共用 _upscale / _imread_u / _imwrite_u / 切字）。
"""
import os, sys
import numpy as np
import cv2
import exp_template_ocr as ET

def _app_base():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

ERDA_TMPL_DIR = os.path.join(_app_base(), "templates_erda")
BADGE_DIR     = os.path.join(_app_base(), "templates_erda_badge")
FIELDS = ["氣息", "碎片", "氣息數量"]
# 各欄位濾鏡門檻 (gray>, sat<)
ERDA_MASK = {"氣息": (165, 70), "碎片": (150, 120), "氣息數量": (175, 95)}


def erda_mask(crop, field="碎片"):
    gth, sth = ERDA_MASK.get(field, (150, 120))
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV); gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return ET._upscale(((gray > gth) & (hsv[:, :, 1] < sth)).astype(np.uint8) * 255)


def _segments(mask):
    cs = np.sum(mask > 0, axis=0); on = np.where(cs > 0)[0]
    if len(on) == 0:
        return mask, []
    mask = mask[:, max(0, on[0] - 4):on[-1] + 5]
    runs = [list(r) for r in ET.TemplateOCR._runs(mask)]
    hs = []
    for a, b in runs:
        rows = np.where(np.any(mask[:, a:b + 1] > 0, axis=1))[0]
        hs.append(rows[-1] - rows[0] + 1 if len(rows) else 0)
    if hs:
        med = np.median([h for h in hs if h > 0]) or 1
        runs = [r for r, h in zip(runs, hs) if h >= 0.55 * med and (r[1] - r[0] + 1) >= 4]
    return mask, runs


def badge_mask(crop):
    gth, sth = ERDA_MASK["氣息數量"]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV); gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mk = ((gray > gth) & (hsv[:, :, 1] < sth)).astype(np.uint8) * 255
    bw = mk > 0; rows = np.any(bw, 1); cols = np.any(bw, 0)
    if rows.any() and cols.any():
        r0, r1 = np.where(rows)[0][[0, -1]]; c0, c1 = np.where(cols)[0][[0, -1]]
        mk = mk[r0:r1 + 1, c0:c1 + 1]
    return cv2.resize(mk, (64, 40), interpolation=cv2.INTER_AREA) if mk.size else np.zeros((40, 64), np.uint8)


class ErdaOCR:
    """氣息/碎片 數字辨識（白字遮罩 + 模板比對）；模板與 EXP 分開存。"""
    CHARSET = list("0123456789/")
    def __init__(self, d=ERDA_TMPL_DIR): self.dir = d; self.tmpl = {}; self.load()
    def _fn(self, ch): return "slash" if ch == "/" else ch
    def load(self):
        self.tmpl = {}
        for ch in self.CHARSET:
            p = os.path.join(self.dir, self._fn(ch) + ".png")
            if os.path.exists(p):
                im = ET._imread_u(p, cv2.IMREAD_GRAYSCALE)
                if im is not None: self.tmpl[ch] = im
    def ready(self): return sum(c.isdigit() for c in self.tmpl) >= 10
    def read(self, crop, field):
        mask, runs = _segments(erda_mask(crop, field))
        if not self.tmpl: return ""
        out = ""
        for a, b in runs:
            seg = ET.TemplateOCR._crop(mask[:, a:b + 1]); sf = seg.astype(np.float32) / 255.0
            best, bs = None, -2.0
            for ch, t in self.tmpl.items():
                tc = ET.TemplateOCR._crop(t)
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
                        samples.setdefault(ch, []).append(ET.TemplateOCR._crop(mask[:, a:b + 1]))
                return True, len(runs)
        return False, len(runs)
    def save(self, samples):
        os.makedirs(self.dir, exist_ok=True)
        for ch, segs in samples.items():
            H = int(np.median([s.shape[0] for s in segs])); W = int(np.median([s.shape[1] for s in segs]))
            rez = [cv2.resize(s, (W, H), interpolation=cv2.INTER_AREA) for s in segs]
            ET._imwrite_u(os.path.join(self.dir, self._fn(ch) + ".png"), np.mean(rez, 0).astype(np.uint8))
        self.load()


class BadgeMatcher:
    """氣息數量 0~20：整張圖比對 21 個範本。範本可由 0~9 數字合成。"""
    def __init__(self, d=BADGE_DIR): self.dir = d; self.refs = {}; self.load()
    def load(self):
        self.refs = {}
        if not os.path.isdir(self.dir): return
        for fn in os.listdir(self.dir):
            if fn.endswith(".png"):
                try: val = int(os.path.splitext(fn)[0])
                except ValueError: continue
                im = ET._imread_u(os.path.join(self.dir, fn), cv2.IMREAD_GRAYSCALE)
                if im is not None: self.refs[val] = cv2.resize(im, (64, 40))
    def ready(self): return len(self.refs) >= 1
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
    def save(self, val, crop):
        rawdir = os.path.join(self.dir, "raw"); os.makedirs(rawdir, exist_ok=True)
        i = 1
        while os.path.exists(os.path.join(rawdir, f"{val}_{i:02d}.png")): i += 1
        ET._imwrite_u(os.path.join(rawdir, f"{val}_{i:02d}.png"), crop)
        masks = []
        for fn in os.listdir(rawdir):
            if fn.startswith(f"{val}_") and fn.endswith(".png"):
                im = ET._imread_u(os.path.join(rawdir, fn), cv2.IMREAD_COLOR)
                if im is not None: masks.append(badge_mask(im).astype(np.float32))
        if masks:
            ET._imwrite_u(os.path.join(self.dir, f"{val}.png"), np.mean(masks, 0).astype(np.uint8))
        self.load(); return i
    def compose_from_digits(self, erda_ocr):
        digs = {c: erda_ocr.tmpl[c] for c in "0123456789" if c in erda_ocr.tmpl}
        miss = [c for c in "0123456789" if c not in digs]
        if miss: return False, miss
        os.makedirs(self.dir, exist_ok=True)
        for V in range(21):
            glyphs = [ET.TemplateOCR._crop(digs[c]) for c in str(V)]
            Hc = max(g.shape[0] for g in glyphs); gap = max(2, int(Hc * 0.12))
            rez = [cv2.resize(g, (max(1, int(g.shape[1] * Hc / g.shape[0])), Hc), interpolation=cv2.INTER_AREA) for g in glyphs]
            tw = sum(r.shape[1] for r in rez) + gap * (len(rez) - 1)
            canvas = np.zeros((Hc, tw), np.uint8); x = 0
            for r in rez:
                canvas[:, x:x + r.shape[1]] = r; x += r.shape[1] + gap
            ET._imwrite_u(os.path.join(self.dir, f"{V}.png"), cv2.resize(canvas, (64, 40), interpolation=cv2.INTER_AREA))
        self.load(); return True, []


def parse_breath(digits):
    d = digits.split("/")[0] if "/" in digits else digits
    if "/" not in digits and d.endswith("1000") and len(d) > 4:
        d = d[:-4]
    try:
        v = int(d)
        return v if 0 <= v <= 1000 else None
    except ValueError:
        return None


class ErdaReader:
    """整合讀取 + 時間平滑。regions = {欄位:(x,y,w,h)}（client 座標）。"""
    def __init__(self):
        self.ocr = ErdaOCR(); self.badge = BadgeMatcher()
        self.breath = None; self.shred = None; self.badge_val = None
        self._b_pending = None; self._b_count = 0
    def configured(self, regions):
        return self.ocr.ready() and regions and all(f in regions for f in FIELDS)
    def reload(self):
        self.ocr.load(); self.badge.load()
    def read(self, client_bgr, regions):
        def crop(f):
            x, y, w, h = regions[f]
            return client_bgr[max(0, y):y + h, max(0, x):x + w]
        # 氣息
        if "氣息" in regions:
            b = parse_breath(self.ocr.read(crop("氣息"), "氣息"))
            if b is not None: self.breath = b
        # 碎片
        if "碎片" in regions:
            s = self.ocr.read(crop("碎片"), "碎片")
            try:
                sv = int(s)
                if 0 <= sv <= 99999999: self.shred = sv
            except ValueError:
                pass
        # 氣息數量：整體比對 + 時間平滑（連續 2 幀同值才改）
        if "氣息數量" in regions and self.badge.refs:
            v, sc = self.badge.read(crop("氣息數量"))
            if v is not None:
                if v == self.badge_val:
                    self._b_pending = None; self._b_count = 0
                elif v == self._b_pending:
                    self._b_count += 1
                    if self._b_count >= 2:
                        self.badge_val = v; self._b_pending = None; self._b_count = 0
                else:
                    self._b_pending = v; self._b_count = 1
        return {"breath": self.breath, "shred": self.shred, "badge": self.badge_val}


if __name__ == "__main__":
    r = ErdaReader()
    print("digits ready:", r.ocr.ready(), "badge refs:", len(r.badge.refs))
