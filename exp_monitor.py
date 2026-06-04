"""
MapleStory EXP Monitor  v5.1
============================
每 N 秒自動擷取 MapleStory 視窗底部，辨識 EXP 百分比。

輸出格式：
  [時間] pct=41.016%  exp=160,668,389,326,359

核心原理：
  1. 截底部 35px 寬條
  2. 用黃色（HSV H=15-45）偵測 EXP 進度條所在行
  3. 只對該行的白色文字跑 OCR，避免 HP/MP 數字干擾
  4. 解析 number[XX.XXX%] 格式

截圖方法（按優先順序）：
  1. mss  — 最穩定
  2. PrintWindow PW_RENDERFULLCONTENT
  3. Desktop DC BitBlt

OCR：Tesseract（若已安裝）→ EasyOCR 備援

需求：
  pip install mss opencv-python pywin32 easyocr

使用方式：
  python exp_monitor.py             正常監控（每 5 秒）
  python exp_monitor.py --interval 3
  python exp_monitor.py --debug     診斷模式，截圖並儲存所有中間結果
"""

import argparse
import ctypes
import os
import re
import time
from datetime import datetime

import cv2
import numpy as np

# 模板比對 OCR（主要辨識器）＋時間聚合層
try:
    from exp_template_ocr import TemplateOCR, ExpTracker
    _HAS_TEMPLATE = True
except Exception:
    _HAS_TEMPLATE = False

# ──────────────────────────────────────────────
# 設定
# ──────────────────────────────────────────────
MONITOR_INTERVAL = 5
EXP_BAR_HEIGHT   = 35      # 底部擷取高度（px）
DEBUG_DIR        = "debug_images"

# EXP 進度條黃色範圍（HSV）
# MapleStory EXP 條是黃-綠漸層，H ≈ 15~65
EXP_YELLOW_LO = np.array([15, 80, 80],  np.uint8)
EXP_YELLOW_HI = np.array([65, 255, 255], np.uint8)
EXP_ROW_MIN_YELLOW_PX = 20   # 至少有這麼多黃色像素才算找到 EXP 條

# 白色文字 HSV 範圍
WHITE_LO = np.array([0,   0,   150], np.uint8)
WHITE_HI = np.array([180, 70,  255], np.uint8)

UPSCALE = 8

TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]
OCR_WHITELIST = "0123456789,.[]%"

OCR_FIXES = str.maketrans({
    'O':'0','o':'0','l':'1','I':'1','|':'1',
    'S':'5','s':'5','B':'8','Z':'2','z':'2',
    'G':'6','g':'9','q':'9','\n':'','\r':'',
})


# ──────────────────────────────────────────────
# DPI
# ──────────────────────────────────────────────

def set_dpi_awareness():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


# ──────────────────────────────────────────────
# 視窗
# ──────────────────────────────────────────────

def find_window():
    """回傳 (hwnd, client_screen_rect) 或 (None, None)。"""
    import win32gui
    found = []

    def cb(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return
        if "MapleStory" not in win32gui.GetWindowText(hwnd):
            return
        r = win32gui.GetWindowRect(hwnd)
        w, h = r[2]-r[0], r[3]-r[1]
        if w >= 800 and h >= 600:
            found.append((hwnd, w*h))

    win32gui.EnumWindows(cb, None)
    if not found:
        return None, None

    found.sort(key=lambda x: x[1], reverse=True)
    hwnd = found[0][0]
    import win32gui
    lt  = win32gui.ClientToScreen(hwnd, (0, 0))
    cr  = win32gui.GetClientRect(hwnd)
    return hwnd, {'left': lt[0], 'top': lt[1],
                  'width': cr[2]-cr[0], 'height': cr[3]-cr[1]}


# ──────────────────────────────────────────────
# 截圖（三種備援）
# ──────────────────────────────────────────────

def _cap_mss(reg):
    try:
        import mss
        x = reg['left']
        y = reg['top'] + reg['height'] - EXP_BAR_HEIGHT
        with mss.mss() as sct:
            raw = sct.grab({"left":x,"top":y,
                            "width":reg['width'],"height":EXP_BAR_HEIGHT})
            img = np.array(raw, dtype=np.uint8)
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR), "mss"
    except Exception as e:
        return None, f"mss:{e}"


def _cap_printwindow(hwnd, reg):
    """
    PrintWindow with PW_RENDERFULLCONTENT=2。
    可以直接讀視窗的渲染 buffer，不受其他視窗遮擋影響。
    注意：DirectX 遊戲常回傳 0 但內容仍然正確，所以不以回傳值判斷成敗，
    改用實際像素值決定是否可用。
    """
    try:
        import win32gui, win32ui
        cw, ch = reg['width'], reg['height']
        hdc  = win32gui.GetWindowDC(hwnd)
        mdc  = win32ui.CreateDCFromHandle(hdc)
        sdc  = mdc.CreateCompatibleDC()
        bmp  = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(mdc, cw, ch)
        sdc.SelectObject(bmp)
        # PW_RENDERFULLCONTENT=2：要求視窗完整渲染（不受遮擋影響）
        # 回傳 0 對 DX 遊戲很常見，不代表失敗，直接看圖片內容
        ctypes.windll.user32.PrintWindow(hwnd, sdc.GetSafeHdc(), 2)
        info = bmp.GetInfo()
        data = bmp.GetBitmapBits(True)
        img  = np.frombuffer(data, np.uint8).reshape((info['bmHeight'],info['bmWidth'],4))
        win32gui.DeleteObject(bmp.GetHandle())
        sdc.DeleteDC(); mdc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hdc)
        full = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return full[ch-EXP_BAR_HEIGHT:, :], "PrintWindow"
    except Exception as e:
        return None, f"PrintWindow:{e}"


def _cap_bitblt(reg):
    try:
        import win32gui, win32ui, win32con
        x = reg['left']
        y = reg['top'] + reg['height'] - EXP_BAR_HEIGHT
        w, h = reg['width'], EXP_BAR_HEIGHT
        hd   = win32gui.GetDesktopWindow()
        dc   = win32gui.GetWindowDC(hd)
        mdc  = win32ui.CreateDCFromHandle(dc)
        sdc  = mdc.CreateCompatibleDC()
        bmp  = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(mdc, w, h)
        sdc.SelectObject(bmp)
        sdc.BitBlt((0,0),(w,h), mdc,(x,y), win32con.SRCCOPY)
        info = bmp.GetInfo()
        data = bmp.GetBitmapBits(True)
        img  = np.frombuffer(data, np.uint8).reshape((info['bmHeight'],info['bmWidth'],4))
        win32gui.DeleteObject(bmp.GetHandle())
        sdc.DeleteDC(); mdc.DeleteDC()
        win32gui.ReleaseDC(hd, dc)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR), "BitBlt"
    except Exception as e:
        return None, f"BitBlt:{e}"


def _has_exp_yellow(img_bgr):
    """
    驗證截圖是否包含 EXP 進度條的黃色像素。
    用來過濾「截到其他視窗」的錯誤結果。
    """
    if img_bgr is None or np.max(img_bgr) < 10:
        return False
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, EXP_YELLOW_LO, EXP_YELLOW_HI)
    return int(np.sum(yellow > 0)) >= EXP_ROW_MIN_YELLOW_PX


def capture_strip(hwnd, reg):
    """
    截底部 EXP_BAR_HEIGHT px，回傳 (img_bgr, method) 或 (None, msg)。

    優先順序根據遊戲視窗是否在前景動態調整：
    - 在前景：mss → PrintWindow → BitBlt
    - 不在前景（被遮擋）：PrintWindow → mss → BitBlt
      PrintWindow 直接讀視窗 buffer，不受其他視窗蓋住影響。
    """
    try:
        import win32gui
        is_focused = (win32gui.GetForegroundWindow() == hwnd)
    except Exception:
        is_focused = True   # 無法判斷時預設 mss 優先

    if is_focused:
        order = [(_cap_mss,(reg,)), (_cap_printwindow,(hwnd,reg)), (_cap_bitblt,(reg,))]
    else:
        order = [(_cap_printwindow,(hwnd,reg)), (_cap_mss,(reg,)), (_cap_bitblt,(reg,))]

    last_err = "所有截圖方法失敗"
    for fn, args in order:
        img, lbl = fn(*args)
        if img is None:
            last_err = lbl
            continue
        if not _has_exp_yellow(img):
            # 截到了但沒有黃色 EXP 條 → 可能截到其他視窗
            last_err = f"{lbl}(無EXP黃色)"
            continue
        return img, lbl

    return None, last_err


# ──────────────────────────────────────────────
# EXP 條定位（黃色偵測）
# ──────────────────────────────────────────────

def find_fill_boundary(text_row_bgr) -> int:
    """
    找出 EXP 填充條的右側邊界（像素列號）。
    填充條左側文字是暗色（反轉後形狀扭曲），右側是白色（正常）。
    回傳填充條右邊界的 x 座標（在 text_row_bgr 的座標系中）。
    """
    hsv = cv2.cvtColor(text_row_bgr, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, EXP_YELLOW_LO, EXP_YELLOW_HI)
    col_has_yellow = np.any(yellow > 0, axis=0)
    if np.any(col_has_yellow):
        return int(np.where(col_has_yellow)[0][-1])
    return 0


def find_exp_text_cols(text_row_bgr, full_w: int):
    """
    在已裁切的文字行中，找出 EXP 文字的水平範圍（相對於 full_w 的絕對座標）。

    策略：
      1. EXP 文字在視窗正中央，先取中間 60% 作為搜尋範圍
      2. 在此範圍內找亮色像素（白字）的水平分布
      3. 以文字邊界 + padding 作為最終裁切範圍

    回傳 (x0, x1) 為相對於 full_w 的絕對像素座標。
    """
    h, w = text_row_bgr.shape[:2]

    # 中間 60%（EXP 文字必在此範圍內）
    cx   = full_w // 2
    lo   = max(0,      cx - int(full_w * 0.30))
    hi   = min(full_w, cx + int(full_w * 0.30))

    # 若傳入的 text_row 已是全寬，直接在此範圍切片分析
    crop = text_row_bgr[:, lo:min(hi, w)]
    if crop.size == 0:
        return lo, hi

    gray     = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # 同時偵測白字（>180，暗色背景）和黑字（<60，黃色背景）
    col_sums = np.sum((gray > 180) | (gray < 60), axis=0)
    cols_on  = np.where(col_sums > 0)[0]

    if len(cols_on) < 4:
        # 找不到足夠文字，退回整個搜尋範圍
        return lo, hi

    pad = max(20, full_w // 80)   # 動態 padding（1920px → ~24px）
    x0  = max(0,      lo + cols_on[0]  - pad)
    x1  = min(full_w, lo + cols_on[-1] + pad + 1)
    return x0, x1


def find_exp_bar_rows(strip_bgr):
    """
    找出 EXP 進度條的行範圍（密集黃色區域）。

    取「最底部」的一段連續密集黃色列作為 EXP 條：
      EXP 進度條永遠在視窗最底邊；上方若有金色活動 UI / 金框背景，
      也會是密集黃色，但會被底部那段以外的群組排除（兩段間有空白列分隔）。
    這樣可避免把上方 HP/MP 與圖示一起切進來造成辨識爆掉。
    """
    hsv      = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2HSV)
    yellow   = cv2.inRange(hsv, EXP_YELLOW_LO, EXP_YELLOW_HI)
    row_sums = np.sum(yellow > 0, axis=1)
    w        = strip_bgr.shape[1]
    h        = strip_bgr.shape[0]

    fill_thresh = max(20, w * 0.10)
    dense       = row_sums >= fill_thresh
    if not dense.any():
        return 0, h

    # 把密集列切成數段（容忍 1 列以內的小空隙），取最底部那一段
    groups = []        # [(y0, y1), ...] 連續密集列，遇到空白列即分段
    y = 0
    while y < h:
        if dense[y]:
            y0 = y
            while y < h and dense[y]:
                y += 1
            groups.append((y0, y - 1))
        else:
            y += 1

    # 取最底部（y1 最大）且高度>=3 的一段；找不到就退回最底部那段
    cand = [g for g in groups if g[1] - g[0] + 1 >= 3]
    by0, by1 = max(cand or groups, key=lambda g: g[1])
    return max(0, by0 - 2), min(h, by1 + 3)


# ──────────────────────────────────────────────
# 前處理
# ──────────────────────────────────────────────

def _upscale(mask):
    """
    8x 放大 + 銳化 + 加粗筆畫。
    加粗（dilate 1px）可以修復細線字如 '1' 被吃掉的問題。
    """
    h, w = mask.shape
    # 放大前補邊，避免邊緣字元被裁切
    padded = cv2.copyMakeBorder(mask, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
    big = cv2.resize(padded,
                     (padded.shape[1] * UPSCALE, padded.shape[0] * UPSCALE),
                     interpolation=cv2.INTER_LANCZOS4)
    _, big = cv2.threshold(big, 127, 255, cv2.THRESH_BINARY)
    # 銳化
    k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    big = cv2.filter2D(big, -1, k)
    _, big = cv2.threshold(big, 127, 255, cv2.THRESH_BINARY)
    # 不加粗：dilate 會填滿 "4" 的孔洞、讓 "%" 變成 X，移除
    return big


def preprocess(row_bgr):
    """
    回傳 [(name, mask_upscaled), ...]，mask 為白字黑底。

    遮罩說明：
    1. YellowAware — 黃色像素處直接反轉灰度（無形態學操作，邊界最乾淨）
    2. Bright      — 亮度閾值 180，捕捉非黃色區域的白字
    3. InvYellow   — 純黃色區域反轉（僅看黃底黑字，補充備援）
    """
    results = []
    hsv  = cv2.cvtColor(row_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(row_bgr, cv2.COLOR_BGR2GRAY)
    k2   = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    # 黃色填充遮罩
    yellow_mask = cv2.inRange(hsv, EXP_YELLOW_LO, EXP_YELLOW_HI)

    # 找填充邊界，清除過渡帶（±20px）
    col_has_yellow = np.any(yellow_mask > 0, axis=0)
    fill_right = int(np.where(col_has_yellow)[0][-1]) if np.any(col_has_yellow) else -1
    bm = 20
    if fill_right >= 0:
        b0 = max(0, fill_right - bm)
        b1 = min(yellow_mask.shape[1], fill_right + bm + 1)
    else:
        b0 = b1 = 0

    # ── Mask 1: YellowAware（支援任何 EXP%）─────────────────────────────────
    # 黃色填充區域：文字是暗色 → 反轉後變白
    # 暗色區域：文字是白色 → 保持不動
    # 這樣無論填充比例多高都能讀到文字
    gray_i16 = gray.astype(np.int16)
    norm = np.where(yellow_mask > 0, 255 - gray_i16, gray_i16)
    norm = np.clip(norm, 0, 255).astype(np.uint8)
    _, ya = cv2.threshold(norm, 160, 255, cv2.THRESH_BINARY)
    # （已移除邊界清除帶：它會吃掉落在填充邊界上的開頭數字，造成「去頭」。
    #   邊界 "|" 假影改由 template OCR 以信心值/寬度剔除。）
    results.append(("YellowAware", _upscale(ya)))

    # ── Mask 2: Bright（暗色區備援）─────────────────────────────────────────
    non_yellow = cv2.bitwise_not(yellow_mask)
    _, mb = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    mb = cv2.bitwise_and(mb, non_yellow)
    # mb[:, b0:b1] = 0  # 同上，移除邊界清除帶
    results.append(("Bright", _upscale(mb)))

    return results


# ──────────────────────────────────────────────
# OCR
# ──────────────────────────────────────────────

_easyocr_reader = None

def _init_easy():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        print("[INFO] 初始化 EasyOCR...")
        _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        print("[INFO] EasyOCR 就緒。")
    return _easyocr_reader


def _setup_tess():
    try:
        import pytesseract
        for p in TESSERACT_PATHS:
            if os.path.isfile(p):
                pytesseract.pytesseract.tesseract_cmd = p
                break
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _ocr_tess(mask):
    import pytesseract
    from PIL import Image
    pil = Image.fromarray(mask)
    # psm 7 = 單行；psm 6 = 統一區塊；psm 11 = 稀疏文字
    best = ""
    for psm in [7, 6, 11]:
        # oem 1 = 純 LSTM，對細線字（如"1"）辨識較準確
        cfg = (f"--psm {psm} --oem 1"
               f" -c tessedit_char_whitelist={OCR_WHITELIST}"
               f" -c load_system_dawg=0 -c load_freq_dawg=0")
        t = pytesseract.image_to_string(pil, config=cfg).strip()
        if len(t) > len(best):
            best = t
        if best and '[' in best and ']' in best:
            break
    return best


def _ocr_easy(mask):
    r = _init_easy()
    res = r.readtext(mask, detail=1, allowlist=OCR_WHITELIST,
                     paragraph=False, min_size=5,
                     text_threshold=0.2, low_text=0.2)
    if not res:
        return ""
    res.sort(key=lambda x: x[0][0][0])
    return " ".join(x[1] for x in res)


def run_ocr(mask, use_tess):
    # Tesseract 優先
    if use_tess:
        try:
            t = _ocr_tess(mask)
            if t: return t, "Tess"
        except Exception:
            pass
    # 備援：EasyOCR
    return _ocr_easy(mask), "Easy"


# ──────────────────────────────────────────────
# 解析
# ──────────────────────────────────────────────

def _pct(s):
    """
    解析百分比，強制 1~3 位小數（楓之谷固定格式如 41.016）。
    - '41.016'   → '41.016'
    - '41.0368'  → '41.036'  (OCR 多讀了 % 的筆畫，截到 3 位)
    - '41016'    → '41.016'  (小數點丟失，自動重建)
    """
    s = re.sub(r'[^\d.]', '', s.replace(',', '.'))
    if not s:
        return None
    if '.' in s:
        parts = s.split('.')
        if len(parts) != 2:
            return None
        int_part = parts[0][:3]          # 整數最多 3 位（0~100）
        dec_part = parts[1][:3]          # 小數截到 3 位
        candidate = f"{int_part}.{dec_part}"
        try:
            v = float(candidate)
            if 0.0 <= v <= 100.0:
                return candidate
        except Exception:
            pass
        return None
    # 無小數點：嘗試插入
    d = re.sub(r'\D', '', s)
    for pos in [2, 1, 3]:
        if pos < len(d):
            c = d[:pos] + '.' + d[pos:pos+3]   # 小數最多 3 位
            try:
                if 0.0 <= float(c) <= 100.0:
                    return c
            except Exception:
                pass
    return None


def _exp(left):
    words = left.split()
    for w in reversed(words):
        r = re.sub(r'[,.]','',w)
        if re.match(r'^\d+$',r) and len(r) >= 8:
            try:
                v = int(r)
                if v >= 1_000_000: return f"{v:,}"
            except: pass
    dw = [re.sub(r'[,.]','',w) for w in words if re.match(r'^[\d,.]+$',w)]
    for s in range(len(dw)):
        for e in range(len(dw),s,-1):
            c = ''.join(dw[s:e])
            if len(c) >= 8:
                try:
                    v = int(c)
                    if v >= 1_000_000: return f"{v:,}"
                except: pass
    return None


def parse(raw):
    if not raw: return None, None
    text = ''.join(OCR_FIXES.get(c,c) for c in raw)

    # ── 精確模式：找 [XX.XXX%] ───────────────────────────────────────────────
    for bm in re.finditer(r'\[\s*([\d.,]{3,8})\s*%?\s*\]', text):
        p = _pct(bm.group(1))
        if p is None: continue
        return _exp(text[:bm.start()].strip()), p

    # ── 半精確：[  被讀成 1 或 11（豎線/填充邊界 artifact）──────────────────
    # 找 1{1,2}XX.XXX%] 或 直接找 XX.XXX%] 格式
    for bm in re.finditer(r'1{0,2}\s*([\d.,]{3,8})\s*%\s*\]', text):
        p = _pct(bm.group(1))
        if p is None: continue
        # EXP 在此 match 之前
        return _exp(text[:bm.start()].strip()), p

    # ── 寬鬆：找緊接 % 的小數（不管有無括號）────────────────────────────────
    # 從右往左找第一個 XX.XXX% 格式（EXP% 必在文字末段）
    for m in reversed(list(re.finditer(r'(\d{1,2}[.,]\d{3})\s*%', text))):
        p = _pct(m.group(1))
        if p is None: continue
        try:
            if 1.0 <= float(p) <= 99.999:
                return _exp(text[:m.start()].strip()), p
        except: pass

    return None, None


# ──────────────────────────────────────────────
# 診斷模式
# ──────────────────────────────────────────────

def run_debug():
    print("=" * 55)
    print("  MapleStory EXP Monitor v5.1 — 診斷模式")
    print("=" * 55)
    set_dpi_awareness()
    os.makedirs(DEBUG_DIR, exist_ok=True)

    hwnd, reg = find_window()
    if hwnd is None:
        print("[✗] 找不到 MapleStory 視窗"); return

    cw, ch = reg['width'], reg['height']
    print(f"[✓] 視窗  {cw}x{ch}  lt=({reg['left']},{reg['top']})")

    # 截底部條
    print("\n── 截圖測試 ──")
    best = None
    for fn, args in [(_cap_mss,(reg,)),(_cap_printwindow,(hwnd,reg)),(_cap_bitblt,(reg,))]:
        img, lbl = fn(*args)
        ok = img is not None and np.max(img) >= 10
        if img is not None:
            mx = int(np.max(img)); mn = float(np.mean(img))
            print(f"  {lbl:22s} max={mx:3d} mean={mn:.1f} {'OK' if ok else '全黑'}")
            cv2.imwrite(os.path.join(DEBUG_DIR, f"dbg_strip_{lbl.split(':')[0]}.png"), img)
            if ok and best is None: best = img
        else:
            print(f"  {lbl}")

    if best is None:
        print("[✗] 截圖全部失敗"); return

    # 黃色定位
    y0, y1 = find_exp_bar_rows(best)
    print(f"\n── EXP 條偵測 ──")
    print(f"  黃色行範圍: y={y0}~{y1}  (原始高度 {best.shape[0]}px)")
    exp_row = best[y0:y1, :]
    cv2.imwrite(os.path.join(DEBUG_DIR, "dbg_exprow.png"), exp_row)
    print(f"  → 已儲存 dbg_exprow.png  shape={exp_row.shape}")

    # 前處理 + OCR
    print("\n── OCR ──")
    use_tess = _setup_tess()
    print(f"  Tesseract: {'✓' if use_tess else '✗ (EasyOCR)'}")
    if not use_tess: _init_easy()

    methods = preprocess(exp_row)
    for name, mask in methods:
        wpx = int(np.sum(mask > 0))
        cv2.imwrite(os.path.join(DEBUG_DIR, f"dbg_mask_{name}.png"), mask)
        raw, eng = run_ocr(mask, use_tess)
        e, p = parse(raw)
        print(f"  {name:6s} white={wpx:7d}  [{eng}] {raw!r:35s}  → pct={p} exp={e}")

    print(f"\n[INFO] 圖片已存至 {os.path.abspath(DEBUG_DIR)}/")


# ──────────────────────────────────────────────
# 正常監控
# ──────────────────────────────────────────────

def run_monitor(interval, exp_digits=0):
    print("=" * 55)
    print("  MapleStory EXP Monitor  v5.1")
    print(f"  間隔={interval}s  Ctrl+C 停止")
    print("=" * 55)
    set_dpi_awareness()
    os.makedirs(DEBUG_DIR, exist_ok=True)

    # 主要辨識器：template OCR（形狀比對）＋ ExpTracker（時間聚合）
    ocr = None
    tracker = None
    if _HAS_TEMPLATE:
        ocr = TemplateOCR()
        if ocr.is_ready():
            tracker = ExpTracker()
            print("[INFO] OCR=Template(shape-match) + ExpTracker")
        else:
            ocr = None
            print("[WARN] 模板未就緒，退回 Tesseract/EasyOCR")

    use_tess = False
    if ocr is None:
        use_tess = _setup_tess()
        if not use_tess:
            _init_easy()
        print(f"[INFO] OCR={'Tesseract' if use_tess else 'EasyOCR'}")

    prev_pct = None

    while True:
        try:
            hwnd, reg = find_window()
            if hwnd is None:
                print(f"[{_ts()}] 找不到視窗..."); time.sleep(interval); continue

            img, cap = capture_strip(hwnd, reg)
            if img is None:
                print(f"[{_ts()}] 截圖失敗 ({cap})"); time.sleep(interval); continue

            y0, y1  = find_exp_bar_rows(img)
            band    = img[y0:y1, :]
            x0, x1  = find_exp_text_cols(band, img.shape[1])
            exp_row = band[:, x0:x1]

            best_e, best_p = None, None
            if ocr is not None:
                r = ocr.recognize_row(band, expected_digits=exp_digits)
                t = tracker.update(r["exp"], r["pct"])
                if t["exp"] is not None:
                    best_e = f"{t['exp']:,}"
                best_p = (f"{t['pct']:.3f}" if t["pct"] is not None else None)
            else:
                for _, mask in preprocess(exp_row):
                    raw, _ = run_ocr(mask, use_tess)
                    if not raw: continue
                    e, p = parse(raw)
                    sc = (1 if p else 0) + (1 if e else 0)
                    bs = (1 if best_p else 0) + (1 if best_e else 0)
                    if sc > bs:
                        best_e, best_p = e, p
                    if best_e and best_p:
                        break
            ts = _ts()
            if best_p:
                diff = ""
                if prev_pct is not None:
                    try: diff = f"  ({float(best_p)-float(prev_pct):+.3f}%)"
                    except Exception: pass
                exp_disp = best_e or "-"
                print(f"[{ts}]  pct={best_p}%  exp={exp_disp}  cap={cap}{diff}")
                prev_pct = best_p
            else:
                ts2 = _ts()
                print(f"[{ts2}] OCR no result [{cap}]")
            time.sleep(interval)
        except KeyboardInterrupt:
            break

def _ts():
    return datetime.now().strftime("%H:%M:%S")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=MONITOR_INTERVAL)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--exp-digits", type=int, default=0,
                    help="已知 EXP 位數（0=自動）；用來剔除填充邊界誤判的多餘數字")
    args = ap.parse_args()
    if args.debug:
        run_debug()
    else:
        run_monitor(args.interval, args.exp_digits)
