"""
exp_template_ocr.py — Template/shape-matching OCR for MapleStory EXP display
============================================================================

為什麼用模板比對而不是 Tesseract：
  EXP 數字疊在黃色經驗條上，隨著經驗增加，黃條邊界會「往右掃過」數字、
  最後甚至掃到百分比。邊界會留下一條細「|」假影，且填充區與暗區交界讓
  Tesseract 把假影讀成多餘的 "1"、或把 5/8、9/3 互換。單幀 Tesseract 幾乎全錯。

  模板比對對這個場景穩很多：
    - "|" 假影配不上任何數字模板（NCC 低）→ 直接剔除，而不是硬猜成數字。
    - 字型固定，等寬，形狀比對即可高準確辨識。

原理：
  1. 取乾淨遮罩（YellowAware，不做會吃掉開頭數字的邊界清除帶）。
  2. 以「列投影空白欄」切出每個字元（run）。
  3. 丟掉開頭的細假影（寬度 <= ARTIFACT_W）。
  4. 以結構錨點（% / [ / ]）把字串切成 EXP 區與 pct 區：
       EXP 區只允許 數字 + 逗號；pct 區只允許 數字 + 小數點。
  5. 每段做 NCC 比對，回傳信心值。

  搭配 ExpTracker（時間層）做逐幀聚合：單調遞增約束 + 高位數多數決 +
  合理性檢查，把單幀殘餘誤差濾掉，輸出穩定值。

模板：templates/ 內每字元一張 PNG（0-9 , . [ ] %），由已驗證的幀建立。
"""

from __future__ import annotations
import os
from pathlib import Path

import cv2
import numpy as np

# ── 設定 ──────────────────────────────────────────────────────────────────────
TEMPLATE_DIR = Path(__file__).parent / "templates"

FILE_CHAR = {
    '0': '0', '1': '1', '2': '2', '3': '3', '4': '4',
    '5': '5', '6': '6', '7': '7', '8': '8', '9': '9',
    'comma': ',', 'dot': '.', 'lbracket': '[', 'rbracket': ']', 'pct': '%',
}

DIGITS = list("0123456789")
ALL_CHARS = list("0123456789,.[]%")

# 遮罩（與建立模板時相同，務必一致）
EXP_YELLOW_LO = np.array([15, 80, 80],  np.uint8)
EXP_YELLOW_HI = np.array([65, 255, 255], np.uint8)
UPSCALE = 8

ARTIFACT_W   = 14    # 開頭寬度 <= 此值的 run 視為邊界假影，丟棄
MIN_RUNS     = 10    # 少於這麼多 run 視為辨識失敗
MIN_CONF     = 0.55  # 單字元 NCC 低於此值視為不可信


# ══════════════════════════════════════════════════════════════════════════════
# 遮罩產生（不做邊界清除帶 → 不會吃掉開頭數字）
# ══════════════════════════════════════════════════════════════════════════════
def _upscale(mask: np.ndarray) -> np.ndarray:
    """8x 放大 + 銳化（與建立模板時相同的處理）。"""
    padded = cv2.copyMakeBorder(mask, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
    big = cv2.resize(padded, (padded.shape[1] * UPSCALE, padded.shape[0] * UPSCALE),
                     interpolation=cv2.INTER_LANCZOS4)
    _, big = cv2.threshold(big, 127, 255, cv2.THRESH_BINARY)
    k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    big = cv2.filter2D(big, -1, k)
    _, big = cv2.threshold(big, 127, 255, cv2.THRESH_BINARY)
    return big


def build_mask(exp_row_bgr: np.ndarray) -> np.ndarray:
    """
    從裁切好的 EXP 文字行（BGR）產生乾淨白字黑底遮罩。
    黃色填充區：文字暗 → 反轉成白；暗區：文字白 → 保留。
    不做邊界清除帶（那會吃掉落在邊界上的開頭數字）。
    """
    hsv  = cv2.cvtColor(exp_row_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(exp_row_bgr, cv2.COLOR_BGR2GRAY)
    ymask = cv2.inRange(hsv, EXP_YELLOW_LO, EXP_YELLOW_HI)
    gi = gray.astype(np.int16)
    norm = np.where(ymask > 0, 255 - gi, gi)
    norm = np.clip(norm, 0, 255).astype(np.uint8)
    _, ya = cv2.threshold(norm, 160, 255, cv2.THRESH_BINARY)
    return _upscale(ya)


# ══════════════════════════════════════════════════════════════════════════════
# 辨識器
# ══════════════════════════════════════════════════════════════════════════════
class TemplateOCR:

    def __init__(self, template_dir: Path | str = TEMPLATE_DIR):
        self.template_dir = Path(template_dir)
        self.templates: dict[str, np.ndarray] = {}
        self._load()

    def _load(self):
        self.templates.clear()
        for fname, ch in FILE_CHAR.items():
            p = self.template_dir / f"{fname}.png"
            if p.exists():
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    self.templates[ch] = img

    def reload(self):
        self._load()

    def is_ready(self) -> bool:
        return sum(c.isdigit() for c in self.templates) >= 10

    # ── 工具 ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _crop(g: np.ndarray) -> np.ndarray:
        bw = g > 127
        rows = np.any(bw, axis=1)
        cols = np.any(bw, axis=0)
        if not rows.any() or not cols.any():
            return g
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        return g[r0:r1 + 1, c0:c1 + 1]

    @staticmethod
    def _runs(mask: np.ndarray) -> list[list[int]]:
        cs = np.sum(mask > 0, axis=0)
        runs = []
        i, n = 0, len(cs)
        while i < n:
            if cs[i] > 0:
                j = i
                while j < n and cs[j] > 0:
                    j += 1
                runs.append([i, j - 1])
                i = j
            else:
                i += 1
        return runs

    def _match(self, seg: np.ndarray, allowed) -> tuple[str | None, float]:
        seg = self._crop(seg)
        if seg.size == 0:
            return None, -1.0
        sf = seg.astype(np.float32) / 255.0
        best, best_s = None, -2.0
        for ch in allowed:
            t = self.templates.get(ch)
            if t is None:
                continue
            tc = self._crop(t)
            th, tw = tc.shape[:2]
            rz = cv2.resize(sf, (tw, th), interpolation=cv2.INTER_AREA)
            a = rz.flatten()
            c = (tc.astype(np.float32) / 255.0).flatten()
            if a.std() < 0.01 or c.std() < 0.01:
                continue
            s = float(np.corrcoef(a, c)[0, 1])
            if s > best_s:
                best_s, best = s, ch
        return best, best_s

    # ── 主辨識 ───────────────────────────────────────────────────────────────
    def recognize(self, mask: np.ndarray) -> dict:
        """
        輸入：乾淨白字黑底遮罩（build_mask 的輸出）。
        回傳 dict：
          exp   : EXP 數字字串（純數字，無逗號）或 None
          pct   : 百分比字串 "DD.DDD" 或 None
          conf  : 平均 NCC 信心值
          ok    : 是否成功解析出 exp 與 pct
        """
        out = {"exp": None, "pct": None, "conf": 0.0, "ok": False}
        # 水平裁切到內容
        cs = np.sum(mask > 0, axis=0)
        on = np.where(cs > 0)[0]
        if len(on) == 0:
            return out
        mask = mask[:, max(0, on[0] - 4):on[-1] + 5]

        runs = self._runs(mask)
        # 丟掉開頭細假影
        while runs and (runs[0][1] - runs[0][0] + 1) <= ARTIFACT_W:
            runs.pop(0)
        if len(runs) < MIN_RUNS:
            return out

        widths = [b - a + 1 for a, b in runs]

        # 錨點：% （最寬、且 NCC 對 % 高）→ 由右往左找
        pct_idx = None
        for i in range(len(runs) - 1, -1, -1):
            ch, sc = self._match(mask[:, runs[i][0]:runs[i][1] + 1], ['%'])
            if (sc > 0.6 and widths[i] >= 45) or widths[i] >= 52:
                pct_idx = i
                break
        if pct_idx is None:
            return out

        # 錨點：[ （在 % 之前找最像 [ 的）
        lb_idx, best_lb = None, -2.0
        for i in range(max(0, pct_idx - 10), pct_idx):
            _, sc = self._match(mask[:, runs[i][0]:runs[i][1] + 1], ['['])
            if sc > best_lb:
                best_lb, lb_idx = sc, i
        if lb_idx is None or lb_idx == 0:
            return out

        confs = []

        # EXP 區：數字 + 逗號
        exp = ""
        for a, b in runs[:lb_idx]:
            ch, sc = self._match(mask[:, a:b + 1], DIGITS + [','])
            confs.append(sc)
            if ch and ch != ',':
                exp += ch

        # pct 區：固定格式，先找小數點（最窄的 run）其餘強制數字
        pct_runs = runs[lb_idx + 1:pct_idx]
        pct = self._parse_pct(mask, pct_runs, confs)

        out["conf"] = float(np.mean(confs)) if confs else 0.0
        out["exp"] = exp if exp.isdigit() and len(exp) >= 6 else None
        out["pct"] = pct
        out["ok"] = out["exp"] is not None and pct is not None
        return out

    def _parse_pct(self, mask, pct_runs, confs) -> str | None:
        """pct 區固定為 N.NNN ~ NN.NNN：小數點是最窄的 run，其餘一律當數字。"""
        if len(pct_runs) < 4:
            return None
        widths = [b - a + 1 for a, b in pct_runs]
        dot_pos = int(np.argmin(widths))           # 最窄者為小數點
        digits = ""
        for k, (a, b) in enumerate(pct_runs):
            if k == dot_pos:
                continue
            ch, sc = self._match(mask[:, a:b + 1], DIGITS)
            confs.append(sc)
            if ch:
                digits += ch
        if len(digits) < 4:
            return None
        # 小數固定 3 位 → 整數位 = 其餘
        int_part = digits[:dot_pos]
        dec_part = digits[dot_pos:dot_pos + 3]
        if not int_part or len(dec_part) < 3:
            return None
        try:
            v = float(f"{int_part}.{dec_part}")
            if 0.0 <= v <= 100.0:
                return f"{int_part}.{dec_part}"
        except ValueError:
            pass
        return None

    def recognize_row(self, exp_row_bgr: np.ndarray) -> dict:
        """便利方法：直接吃裁切好的 EXP 文字行（BGR）。"""
        return self.recognize(build_mask(exp_row_bgr))


# ══════════════════════════════════════════════════════════════════════════════
# 時間聚合層 —— 單調遞增約束 + 高位數多數決 + 合理性檢查
# ══════════════════════════════════════════════════════════════════════════════
class ExpTracker:
    """
    把逐幀（可能含少量誤差）的辨識結果聚合成穩定、單調遞增的 EXP / pct。

    設計目標：低延遲（好幀直接輸出真值）+ 對單幀誤差免疫。
      1. 位數過濾：與近期眾數位數不符的讀數丟棄（多/少切一位）。
      2. 自適應增幅上限：用近期實際增量估「每幀合理增幅」，
         不必為不同抓取間隔手調門檻。
      3. 單調 + 上限：本幀 >= 上次輸出，且增幅在合理範圍 → 直接採用（零延遲、真值）。
      4. 否則維持上次good值（監控場景可接受短暫保持），等下一幀好讀數。
      5. pct 同樣單調 + 合理跳動過濾。
    """

    def __init__(self, window: int = 8, gain_k: float = 8.0,
                 abs_max_gain: float = 5e10):
        self.window = window          # 估計增幅用的歷史視窗
        self.gain_k = gain_k          # 容許 = gain_k * 近期中位增幅
        self.abs_max_gain = abs_max_gain
        self.exp_hist: list[int] = []     # 近期被接受的 EXP
        self.deltas: list[int] = []       # 近期被接受的相鄰增量
        self.last_exp: int | None = None
        self.last_pct: float | None = None

    def _expected_len(self) -> int | None:
        if not self.exp_hist:
            return None
        lens = [len(str(v)) for v in self.exp_hist]
        return max(set(lens), key=lens.count)

    def _gain_limit(self) -> float:
        if self.deltas:
            med = sorted(self.deltas)[len(self.deltas)//2]
            med = max(med, 1)
            return min(self.abs_max_gain, max(self.gain_k * med, 2e9))
        return self.abs_max_gain     # 尚無歷史增量時放寬

    def update(self, exp_str, pct_str) -> dict:
        """
        丟一幀辨識結果進來，回傳：
          {"exp": int|None, "pct": float|None, "raw_exp": int|None,
           "accepted": bool, "reason": str}
        accepted=True 表示本幀被採用為新輸出；False 表示維持上次值。
        """
        res = {"exp": self.last_exp, "pct": self.last_pct,
               "raw_exp": None, "accepted": False, "reason": ""}

        raw_exp = int(exp_str) if (exp_str and exp_str.isdigit()) else None
        raw_pct = None
        if pct_str:
            try:
                v = float(pct_str)
                if 0.0 <= v <= 100.0:
                    raw_pct = v
            except ValueError:
                pass
        res["raw_exp"] = raw_exp

        if raw_exp is None:
            res["reason"] = "no_exp"
            return res

        # 位數過濾
        exp_len = self._expected_len()
        if exp_len is not None and len(str(raw_exp)) != exp_len:
            res["reason"] = f"len{len(str(raw_exp))}!={exp_len}"
            return res

        # 首筆：直接接受
        if self.last_exp is None:
            self._accept(raw_exp, raw_pct, delta=None)
            res.update(exp=self.last_exp, pct=self.last_pct,
                       accepted=True, reason="first")
            return res

        delta = raw_exp - self.last_exp
        if delta < 0:
            res["reason"] = "decrease_reject"
            return res
        if delta > self._gain_limit():
            res["reason"] = f"gain_reject({delta:.2e})"
            return res

        # pct 合理性（容忍 OCR 抖動 0.01）
        if raw_pct is not None and self.last_pct is not None and raw_pct < self.last_pct - 0.01:
            raw_pct = None   # pct 不可信就沿用舊 pct，但 EXP 仍可採用

        self._accept(raw_exp, raw_pct, delta=delta)
        res.update(exp=self.last_exp, pct=self.last_pct,
                   accepted=True, reason="ok")
        return res

    def _accept(self, exp_int, pct_val, delta):
        self.last_exp = exp_int
        if pct_val is not None:
            self.last_pct = pct_val
        self.exp_hist.append(exp_int)
        if len(self.exp_hist) > self.window:
            self.exp_hist.pop(0)
        if delta is not None:
            self.deltas.append(delta)
            if len(self.deltas) > self.window:
                self.deltas.pop(0)


if __name__ == "__main__":
    ocr = TemplateOCR()
    print(f"Ready: {ocr.is_ready()}  chars: {sorted(ocr.templates)}")
