@echo off
echo ============================================================
echo  小鲲桌面宠物 — Python pywebview 版本
echo ============================================================
echo.
echo 首次使用需要安装: pip install pywebview
echo.
set URL=http://192.168.10.8:8765/companion?surface=desktop
python -c "import webview; webview.create_window('小鲲桌宠','%URL%',width=360,height=500,resizable=True,frameless=False,on_top=True); webview.start(gui='edgechromium')"
pause
