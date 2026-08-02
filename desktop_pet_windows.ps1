# ============================================================
# 小鲲桌面宠物 — Windows 一键启动（Chrome/Edge 无边框窗口）
# ============================================================
# 用法: 右键 → "使用 PowerShell 运行"
# 或者在 PowerShell 中执行:
#   powershell -ExecutionPolicy Bypass -File desktop_pet_windows.ps1
# ============================================================

$url = "http://192.168.10.8:8765/companion?surface=desktop"
$width = 360
$height = 500
$x = [System.Windows.Forms.Screen]::PrimaryScreen.WorkingArea.Width - $width - 20
$y = 50

# 找 Chrome 或 Edge
$browsers = @(
    "C:\Program Files\Google\Chrome\Application\chrome.exe",
    "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "${env:LOCALAPPDATA}\Google\Chrome\Application\chrome.exe",
    "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "C:\Program Files\Microsoft\Edge\Application\msedge.exe"
)

$browser = $null
foreach ($b in $browsers) {
    if (Test-Path $b) { $browser = $b; break }
}

if (-not $browser) {
    Write-Host "未找到 Chrome 或 Edge，请安装后重试"
    Read-Host
    exit 1
}

Write-Host "启动小鲲桌宠..."
Write-Host "URL: $url"
Write-Host "浏览器: $browser"

# 关闭旧窗口
taskkill /F /IM chrome.exe /FI "WINDOWTITLE eq 小鲲*" 2>$null

Start-Process -FilePath $browser -ArgumentList @(
    "--app=$url",
    "--window-size=$width,$height",
    "--window-position=$x,$y",
    "--disable-infobars",
    "--disable-session-crashed-bubble",
    "--no-first-run",
    "--new-window"
)

Write-Host "✅ 小鲲桌宠已启动！关闭此窗口即可。"
