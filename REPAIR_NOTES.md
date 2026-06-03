# EXPmonitor 修復筆記（2026-06-03）

## 一句話
擷取與定位都正常；問題出在「辨識」。已從 Tesseract 換成**形狀模板比對 + 時間聚合**，
離線在你的 20 張存檔上達成 100% 可讀、輸出全程單調遞增、3/3 人工正解相符。

## 原本壞在哪
1. **去頭（最致命）**：`exp_monitor.py` 的 `preprocess()` 會在黃色填充邊界
   清掉 ±20px（共 41px）。EXP 數字置中、左緣剛好壓在邊界上，於是開頭 1~2 位
   數字被擦掉（`177,960,...` → `.960,...`）。→ 已移除該清除帶。
2. **Tesseract 不可靠**：字太小，`5/8`、`9/3` 互換，邊界 `|` 假影被讀成多餘的 `1`。
   單幀 20/20 全錯。
3. **舊模板是錯的**：`templates/` 是用「被汙染的 CSV 標籤」訓練出來的，
   自信地輸出錯誤。→ 已用人工核對過的幀重建。

## 現在怎麼運作
- `exp_template_ocr.py`（新檔）
  - `build_mask()`：黃底反轉產生乾淨白字遮罩（**不做會去頭的邊界清除**）。
  - `TemplateOCR`：列投影切字 → 丟掉開頭細假影 → 用 `% [ ]` 當錨點把字串切成
    「EXP 區（只允許數字+逗號）」與「pct 區（只允許數字+小數點）」→ 每段做 NCC 形狀比對。
    `|` 假影配不上任何數字，自然被排除，而不是硬猜成 `1`。
  - `ExpTracker`：時間層。位數過濾 + 自適應增幅上限 + 單調約束。
    好幀直接輸出真值（零延遲），單幀錯誤（如某幀把 9 讀成 4）會被擋下並維持上一個好值。
- `exp_monitor.py`（主控台版，`RUN.BAT`）：已接上 TemplateOCR + ExpTracker。
- `exp_monitor_qt.py`（Qt UI）：worker 改用 TemplateOCR；UI 原本的 max_exp×pct
  驗證層保留不動（它本來就設計來吃乾淨的 pct）。
- 模板不就緒時，兩者都會自動退回舊的 Tesseract/EasyOCR。

## 怎麼用 / 怎麼驗證
- 跑監控：`python exp_monitor.py`（或 Qt：`python exp_monitor_qt.py`）
- 回歸測試（改完程式必跑）：`python selftest_ocr.py`
  - 應看到「可讀率 20/20、單調 OK、正解 3/3」。

## 之後若又不準（字體/解析度/視窗大小變了）
模板是綁定目前字體大小的。換環境後請：
1. 用 `python exp_monitor_debug.py` 錄一段新的 debug session。
2. 人工核對幾張的正確數字，更新 `selftest_ocr.py` 的 `KNOWN_GT`。
3. 需要的話用該 session 重建模板（見 `build_templates.py` / `exp_ocr_ml.py`）。

## 安全網
先在 Windows 上跑一次 `SETUP_GIT.bat` 建立 git。之後每次要存檔點：
`git add -A && git commit -m "說明"`；要還原：`git checkout -- .`。
