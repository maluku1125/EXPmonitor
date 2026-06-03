"""
exp_ocr_ml.py — Template-based OCR for MapleStory EXP display
==============================================================

原理：
  1. 列投影分割（column projection）：找出各字元的左右邊界
  2. 正規化互相關（NCC）比對：每個字元段與模板比對，取最高分
  3. 重建字串

字元集：0-9  ,  .  [  ]  %
模板儲存：templates/ 資料夾，每個字元一個 PNG

建立模板：
  python build_templates.py debug_20260603_XXXXXX/

使用：
  from exp_ocr_ml import TemplateOCR
  ocr = TemplateOCR()
  if ocr.is_ready():
      text = ocr.recognize(binary_mask_image)
"""

import cv2
import json
import numpy as np
from pathlib import Path

# ── 設定 ─────────────────────────────────────────────────────────────────────
TEMPLATE_DIR = Path(__file__).parent / "templates"

# 字元類別 → 檔案名稱對應
CHAR_FILE = {
    '0': '0', '1': '1', '2': '2', '3': '3', '4': '4',
    '5': '5', '6': '6', '7': '7', '8': '8', '9': '9',
    ',': 'comma', '.': 'dot',
    '[': 'lbracket', ']': 'rbracket', '%': 'pct',
}
FILE_CHAR = {v: k for k, v in CHAR_FILE.items()}

# 分割參數
MIN_SEG_WIDTH   = 12   # 最小有效字元寬（像素）；逗號/點太小會被過濾，只留數字和括號
GAP_THRESHOLD   = 4    # 連續空白列數視為字元間隔
MIN_NCC_SCORE   = 0.5  # 低於此分數視為無效比對


# ══════════════════════════════════════════════════════════════════════════════
# 核心類別
# ══════════════════════════════════════════════════════════════════════════════
class TemplateOCR:

    def __init__(self, template_dir: Path = TEMPLATE_DIR):
        self.template_dir = Path(template_dir)
        self.templates: dict[str, np.ndarray] = {}
        self._load()

    # ── 模板管理 ─────────────────────────────────────────────────────────────

    def _load(self):
        """從磁碟載入所有模板。"""
        self.templates.clear()
        for fname, char in FILE_CHAR.items():
            p = self.template_dir / f"{fname}.png"
            if p.exists():
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    self.templates[char] = img

    def reload(self):
        self._load()

    def is_ready(self) -> bool:
        """至少有 10 個數字模板才算就緒。"""
        digits = [c for c in self.templates if c.isdigit()]
        return len(digits) >= 10

    def save_template(self, char: str, img: np.ndarray):
        """儲存單一字元的模板圖像。"""
        self.template_dir.mkdir(parents=True, exist_ok=True)
        fname = CHAR_FILE.get(char)
        if fname is None:
            return
        path = self.template_dir / f"{fname}.png"
        cv2.imwrite(str(path), img)
        self.templates[char] = img

    # ── 字元分割（列投影）────────────────────────────────────────────────────

    def segment(self, binary_img: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
        """
        以列投影（column sum）分割字元。
        回傳 [(x_start, x_end, char_img), ...]，由左到右排序。
        """
        if binary_img.ndim == 3:
            binary_img = cv2.cvtColor(binary_img, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(binary_img, 127, 255, cv2.THRESH_BINARY)

        col_sums = np.sum(bw > 0, axis=0)

        segments: list[tuple[int, int, np.ndarray]] = []
        in_char   = False
        x_start   = 0
        gap_count = 0

        for x, val in enumerate(col_sums):
            if val > 0:
                if not in_char:
                    x_start  = x
                    in_char  = True
                gap_count = 0
            else:
                if in_char:
                    gap_count += 1
                    if gap_count >= GAP_THRESHOLD:
                        seg_x1 = x - gap_count
                        seg    = bw[:, x_start:seg_x1]
                        if seg.shape[1] >= MIN_SEG_WIDTH:
                            segments.append((x_start, seg_x1, seg))
                        in_char   = False
                        gap_count = 0

        if in_char:
            seg = bw[:, x_start:]
            if seg.shape[1] >= MIN_SEG_WIDTH:
                segments.append((x_start, bw.shape[1], seg))

        return segments


    # ── 工具：裁切到有效像素邊界框 ───────────────────────────────────────────

    @staticmethod
    def _crop_content(img):
        bw = (img > 127)
        rows = np.any(bw, axis=1)
        cols = np.any(bw, axis=0)
        if not rows.any() or not cols.any():
            return img
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return img[rmin:rmax+1, cmin:cmax+1]

    # ── 單字元比對（NCC）
    # ── 單字元比對（NCC）─────────────────────────────────────────────────────

    def match_char(self, seg_img: np.ndarray) -> tuple[str | None, float]:
        """
        將字元圖像與所有模板做 NCC 比對，回傳 (最佳字元, 分數)。
        """
        if not self.templates or seg_img.size == 0:
            return None, 0.0

        # 裁切到有效內容區域，排除四周黑色邊框
        seg_c = self._crop_content(seg_img)
        seg_f = seg_c.astype(np.float32) / 255.0

        best_char  = None
        best_score = -2.0

        for char, tmpl in self.templates.items():
            tmpl_c = self._crop_content(tmpl)
            th, tw = tmpl_c.shape[:2]
            # 縮放到模板（裁切後）尺寸
            resized = cv2.resize(seg_f, (tw, th), interpolation=cv2.INTER_AREA)
            t_f     = tmpl_c.astype(np.float32) / 255.0

            r_flat = resized.flatten()
            t_flat = t_f.flatten()

            if r_flat.std() < 0.01 or t_flat.std() < 0.01:
                # 全白或全黑，跳過
                continue

            score = float(np.corrcoef(r_flat, t_flat)[0, 1])
            if score > best_score:
                best_score = score
                best_char  = char

        return best_char, best_score

    # ── 完整辨識 ─────────────────────────────────────────────────────────────

    def _digit_segments(self, binary_img):
        """只取數字寬度的 segment（排除逗號、點等小符號）。"""
        segs = self.segment(binary_img)
        return [(x0, x1, s) for x0, x1, s in segs if 30 <= s.shape[1] <= 70]

    def recognize(self, binary_img, min_score=MIN_NCC_SCORE):
        """
        辨識數字序列，回傳字串（只含數字，跳過分隔符號）。
        """
        segs   = self._digit_segments(binary_img)
        result = ""
        for x0, x1, seg in segs:
            char, score = self.match_char(seg)
            if char is not None and score >= min_score:
                result += char
            else:
                result += "?"
        return result

    def recognize_with_scores(self, binary_img, min_score=MIN_NCC_SCORE):
        """同 recognize，同時回傳每個數字的分數。"""
        segs   = self._digit_segments(binary_img)
        result = ""
        scores = []
        for x0, x1, seg in segs:
            char, score = self.match_char(seg)
            scores.append(score)
            result += (char if char and score >= min_score else "?")
        return result, scores


# ══════════════════════════════════════════════════════════════════════════════
# 模板建立工具函式（由 build_templates.py 呼叫）
# ══════════════════════════════════════════════════════════════════════════════

def build_templates_from_debug(debug_dir: str | Path,
                                template_dir: str | Path = TEMPLATE_DIR,
                                min_frames: int = 5) -> int:
    """
    從 debug session 的 CSV 和處理後圖像自動建立模板。

    流程：
      1. 讀 session.csv，找到通過驗證且有正確 EXP 值的幀
      2. 對每幀，用 Combined → Bright → HSV 順序嘗試載入遮罩圖
      3. 以列投影分割字元，對比預期字串（exp + pct）
      4. 若長度吻合，把每個字元存入對應 class 的樣本列表
      5. 對每個 class 取平均作為模板儲存

    回傳成功建立的模板數量。
    """
    import csv as csv_mod

    debug_dir    = Path(debug_dir)
    template_dir = Path(template_dir)
    csv_path     = debug_dir / "session.csv"

    if not csv_path.exists():
        print(f"[ERROR] session.csv 不存在：{csv_path}")
        return 0

    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv_mod.DictReader(f))

    ocr = TemplateOCR(template_dir)
    char_samples: dict[str, list[np.ndarray]] = {c: [] for c in CHAR_FILE}

    print(f"共 {len(rows)} 幀，開始抽取...")

    # 策略：MIN_SEG_WIDTH=12 會過濾掉逗號/點，只留數字和括號
    # 所以 expected_digits 只取 exp_str 的數字部分（去掉逗號）
    # 以「純數字序列」做分割比對，可靠性更高

    ok_frames = 0
    for row in rows:
        if row.get("status") != "ok":
            continue

        # CSV 欄位（新版 debug）：pct = 百分比，exp = EXP 整數
        # （舊版相反，這裡兩種都嘗試）
        pct_str = row.get("pct", "").strip()
        exp_str = row.get("exp", "").strip()
        # 若像舊版（pct 欄存大數字），嘗試交換
        try:
            pct_f = float(pct_str)
            if pct_f > 100:   # pct_str 其實是 EXP，需交換
                pct_str, exp_str = exp_str, pct_str
                pct_f = float(pct_str)
        except Exception:
            try:
                pct_str, exp_str = exp_str, pct_str
                pct_f = float(pct_str)
            except Exception:
                continue
        if not exp_str or pct_f <= 0 or pct_f >= 100:
            continue
        try:
            int(exp_str.replace(",", ""))
        except Exception:
            continue

        # 只取數字序列（逗號/點在高 threshold 下被過濾，只留數字）
        digits_only = (exp_str.replace(",", "")        # EXP 數字
                       + pct_str.replace(".", ""))     # pct 數字（去掉小數點）
        # e.g. "174591386235645" + "44570" = "17459138623564544570"

        frame = row.get("frame", "").strip()
        if not frame:
            continue

        loaded = False
        for mask_name in ["YellowAware", "Combined", "Bright", "HSV", "InvYellow"]:
            img_path = debug_dir / f"{int(frame):05d}_{mask_name}.png"
            if not img_path.exists():
                continue
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            segs = ocr.segment(img)
            # 只保留數字寬度的片段（排除逗號/點 ≤20px，也排除過寬的合體字元）
            # 數字在 8x upscale 下通常 35-55px 寬
            digit_segs = [(x0, x1, s) for x0, x1, s in segs
                          if 30 <= s.shape[1] <= 70]

            if len(digit_segs) == 0:
                continue

            # 比對長度：digit_segs 應對應 digits_only 的字元數
            n_segs = len(digit_segs)
            n_exp  = len(digits_only)

            # 只接受完全吻合（精確對位才能正確標記每個數字）
            if n_segs != n_exp:
                continue

            for i in range(n_segs):
                x0, x1, seg_img = digit_segs[i]
                char = digits_only[i]
                if char.isdigit() and char in char_samples:
                    cropped = ocr._crop_content(seg_img)
                    if cropped.size > 0:
                        char_samples[char].append(cropped.copy())

            ok_frames += 1
            loaded = True
            break


        if ok_frames % 10 == 0 and ok_frames > 0:
            print(f"  已處理 {ok_frames} 幀...")

    print(f"\n共收集到 {ok_frames} 幀的字元樣本")

    # 建立並儲存模板
    template_dir.mkdir(parents=True, exist_ok=True)
    built = 0
    for char, samples in char_samples.items():
        if len(samples) < min_frames:
            print(f"  '{char}': 樣本不足（{len(samples)}<{min_frames}），跳過")
            continue

        # 統一大小（取中位數寬高）
        heights = [s.shape[0] for s in samples]
        widths  = [s.shape[1] for s in samples]
        med_h   = int(np.median(heights))
        med_w   = int(np.median(widths))

        resized = [cv2.resize(s, (med_w, med_h), interpolation=cv2.INTER_AREA)
                   for s in samples]
        tmpl    = np.mean(resized, axis=0).astype(np.uint8)

        fname = CHAR_FILE[char]
        out   = template_dir / f"{fname}.png"
        cv2.imwrite(str(out), tmpl)
        print(f"  {char!r}: {len(samples)} samples -> {out.name}  ({med_w}x{med_h})")
        built += 1

    with open(template_dir / "meta.json", "w") as f:
        json.dump({"built": built}, f)

    print(f"Done! Built {built} templates -> {template_dir}")
    return built


if __name__ == "__main__":
    ocr = TemplateOCR()
    print(f"Ready: {ocr.is_ready()}  Loaded: {sorted(ocr.templates.keys())}")
