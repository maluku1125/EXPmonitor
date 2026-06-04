@echo off
rem ============================================
rem  EXPMonitor slack trigger : Rick Roll
rem  偷懶偵測觸發時開啟 Never Gonna Give You Up
rem  可用的環境變數: %SLACK_SECONDS%  %SLACK_EXP%  %SLACK_PCT%
rem ============================================
echo Never gonna give you up...  (slacked %SLACK_SECONDS%s)
start "" "https://www.youtube.com/watch?v=dQw4w9WgXcQ&autoplay=1"
