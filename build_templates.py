"""
build_templates.py — 從 debug session 自動建立模板庫
=====================================================

使用方式：
  python build_templates.py debug_20260603_210547
  python build_templates.py debug_20260603_210547 --min-frames 3
  python build_templates.py debug_20260603_210547 --out templates_v2

建立完成後，可用 --test 確認效果：
  python build_templates.py debug_20260603_210547 --test
"""

import argparse
from pathlib import Path
from exp_ocr_ml import TemplateOCR, build_templates_from_debug, TEMPLATE_DIR
import cv2, numpy as np


def test_templates(debug_dir: Path, template_dir: Path):
    """對 debug session 中所有 ok 幀跑一次模板 OCR，印出對比結果。"""
    import csv

    csv_path = debug_dir / "session.csv"
    if not csv_path.exists():
        print("找不到 session.csv")
        return

    ocr = TemplateOCR(template_dir)
    if not ocr.is_ready():
        print("模板不完整，請先建立")
        return

    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    print(f"\n{'─'*70}")
    print(f"{'幀':>5}  {'預期 EXP':20}  {'預期 %':8}  {'ML結果':35}  {'分數'}")
    print(f"{'─'*70}")

    ok = fail = 0
    for row in rows[:50]:   # 只測前 50 幀
        if row.get("status") != "ok":
            continue
        exp_str = row.get("pct", "").strip()
        pct_str = row.get("exp", "").strip()
        frame   = row.get("frame", "").strip()
        if not exp_str or not pct_str or not frame:
            continue

        expected = f"{exp_str}[{pct_str}%]"

        # 嘗試 Combined 遮罩
        img_path = debug_dir / f"{int(frame):05d}_Combined.png"
        if not img_path.exists():
            img_path = debug_dir / f"{int(frame):05d}_Bright.png"
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        result, scores = ocr.recognize_with_scores(img)
        avg_score = float(np.mean(scores)) if scores else 0.0
        # 只比數字序列（template OCR 只識別數字）
        exp_digits = expected.replace(",","").replace("[","").replace("]","").replace("%","").replace(".","")
        res_digits = result.replace("?","")
        match = "OK" if exp_digits.startswith(res_digits) and len(res_digits) >= len(exp_digits)-2 else "NG"
        if match == "OK":
            ok += 1
        else:
            fail += 1
        print(f"  {frame:>5}  {exp_str:20}  {pct_str:8}  {result:35}  {avg_score:.3f}  {match}")

    total = ok + fail
    print(f"{'─'*70}")
    print(f"  結果：{ok}/{total} 正確 ({ok/total*100:.1f}%)" if total else "  無資料")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("debug_dir", help="debug session 資料夾路徑")
    ap.add_argument("--out",        default=str(TEMPLATE_DIR),
                    help=f"模板輸出目錄（預設 {TEMPLATE_DIR}）")
    ap.add_argument("--min-frames", type=int, default=5,
                    help="每個字元最少需要幾個樣本（預設 5）")
    ap.add_argument("--test",       action="store_true",
                    help="建立完成後立即測試精度")
    args = ap.parse_args()

    debug_dir    = Path(args.debug_dir)
    template_dir = Path(args.out)

    if not debug_dir.exists():
        print(f"[ERROR] 資料夾不存在：{debug_dir}")
        return

    print(f"Debug dir : {debug_dir}")
    print(f"Template  : {template_dir}")
    print(f"Min frames: {args.min_frames}\n")

    n = build_templates_from_debug(debug_dir, template_dir, args.min_frames)

    if n > 0 and args.test:
        test_templates(debug_dir, template_dir)


if __name__ == "__main__":
    main()
