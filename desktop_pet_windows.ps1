# ============================================================
# 小鲲桌面宠物 — Windows 一键启动（Chrome/Edge 无边框窗口）
# ============================================================
# 用法: 右键 → "使用 PowerShell 运行"
# 或者在 PowerShell 中执行:
#   powershell -ExecutionPolicy Bypass -File desktop_pet_windows.ps1
# ============================================================

$url = "http://192.168.10.8:8765/companion?surface=desktop"
$width = 360
$height = 520

# 计算窗口位置（右下角），兜底默认值
try {
    Add-Type -AssemblyName System.Windows.Forms
    $screen = [System.Windows.Forms.Screen]::PrimaryScreen.WorkingArea
    $x = $screen.Width - $width - 20
    $y = $screen.Height - $height - 60
} catch {
    $x = 1520; $y = 100  # 常见 1080p 显示器右下角
}

# 找 Chrome / Edge / Chromium
$browser = $null
$browsers = @(
    "C:\Program Files\Google\Chrome\Application\chrome.exe",
    "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "$env:LOCALAPPDATA\Google\Chrome\Application\chrome.exe",
    "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe"
)
foreach ($b in $browsers) {
    if (Test-Path $b) { $browser = $b; break }
}
if (-not $browser) {
    Write-Host "未找到 Chrome/Edge，请安装后重试。或者直接用浏览器打开: $url"
    Read-Host; exit 1
}

# 关闭旧桌宠窗口
taskkill /F /FI "WINDOWTITLE eq 小鲲*" 2>$null

Write-Host "小鲲桌宠启动中..."
Write-Host "URL: $url"
Write-Host "窗口: ${width}x${height}  @ ($x, $y)"

Start-Process -FilePath $browser -ArgumentList @(
    "--app=$url",
    "--window-size=$width,$height",
    "--window-position=$x,$y",
    "--disable-infobars",
    "--disable-session-crashed-bubble",
    "--no-first-run",
    "--new-window"
)

Write-Host "✅ 小鲲桌宠已启动！"
