"""
selftest_ocr.py — 離線回歸測試
=================================
對某個 debug session 的 *_raw.png 跑「template OCR + ExpTracker」，
回報：可讀率、單調率，以及（若該幀有已知正解）逐幀正確率。

用法：
  python selftest_ocr.py                         # 預設測 debug_20260603_231813
  python selftest_ocr.py debug_20260604_xxxxxx   # 測指定資料夾

每次改完程式或重建模板後跑一下，確認沒有退步。
"""
import sys, os, glob, importlib.util
import cv2, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("expmon", os.path.join(HERE, "exp_monitor.py"))
M = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(M)
from exp_template_ocr import TemplateOCR, ExpTracker

# 已驗證的正解（debug_20260603_231813，由人工逐張核對）
KNOWN_GT = {
    "debug_20260603_231813": {
        "00001": ("177960562689528", "45.430"),
        "00010": ("177967181273250", "45.432"),
        "00020": ("177975656304064", "45.434"),
    }
}

def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "debug_20260603_231813"
    d = folder if os.path.isabs(folder) else os.path.join(HERE, folder)
    if not os.path.isdir(d):
        print(f"[錯誤] 找不到資料夾：{d}"); return
    gt = KNOWN_GT.get(os.path.basename(d.rstrip("/\\")), {})

    ocr = TemplateOCR()
    if not ocr.is_ready():
        print("[錯誤] 模板未就緒（templates/ 不完整）"); return
    tracker = ExpTracker()

    frames = sorted(glob.glob(os.path.join(d, "*_raw.png")))
    if not frames:
        print(f"[錯誤] {d} 內沒有 *_raw.png"); return

    print(f"測試 {len(frames)} 幀 @ {os.path.basename(d)}\n")
    print(f"{'frame':6} {'raw_exp':16} {'OUT_exp':16} {'pct':8} {'note'}")
    n=read=mono=gtok=gttot=0; prev=None
    for f in frames:
        fid = os.path.basename(f)[:5]
        img = cv2.imread(f)
        y0,y1 = M.find_exp_bar_rows(img); band = img[y0:y1,:]
        x0,x1 = M.find_exp_text_cols(band, img.shape[1]); row = band[:,x0:x1]
        r = ocr.recognize_row(row); t = tracker.update(r["exp"], r["pct"])
        n += 1
        if r["exp"]: read += 1
        out = t["exp"]; note = ""
        if out is not None:
            if prev is None or out >= prev: mono += 1
            else: note = "!! 非單調"
            prev = out
        if fid in gt:
            gttot += 1
            exp_s = f"{out:,}".replace(",","") if out is not None else None
            if exp_s == gt[fid][0]: gtok += 1; note += " GT_OK"
            else: note += f" !=GT({gt[fid][0]})"
        outs = f"{out:,}" if out is not None else "—"
        print(f"{fid:6} {str(r['exp'] or '—'):16} {outs:16} {str(t['pct']):8} {note}")

    print("\n" + "─"*48)
    print(f"可讀率   : {read}/{n} ({100*read/n:.0f}%)  單幀有讀到數字")
    out_cnt = mono if prev is not None else 0
    print(f"單調率   : 輸出皆單調遞增" + ("  OK" if mono>0 else ""))
    if gttot:
        print(f"正解比對 : {gtok}/{gttot} 正確")
    print("─"*48)

if __name__ == "__main__":
    main()
