<#
VoiceNote 一键环境准备。

用法（在项目根目录）：
    powershell -ExecutionPolicy Bypass -File scripts\setup.ps1

做完这几件事：建虚拟环境 → 装依赖 → 生成 config.toml → 体检。

注意：首次启动时 faster-whisper 会自己下载约 1.6GB 模型权重，这一步不在
本脚本里（放在首次运行时做，避免脚本跑很久且失败后不好重试）。模型缓存
默认在 %USERPROFILE%\.cache\huggingface。国内网络拉不动的话，先设
    $env:HF_ENDPOINT = "https://hf-mirror.com"
再启动即可。
#>

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Step($text) { Write-Host "==> $text" -ForegroundColor Cyan }

Step "检查 uv"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "找不到 uv。请先安装：https://docs.astral.sh/uv/getting-started/installation/" -ForegroundColor Red
    exit 1
}
uv --version

Step "创建虚拟环境（Python 3.13）"
uv venv --python 3.13

Step "安装依赖"
# -e 装成本地可编辑包，这样 python -m voicenote 在任何目录都能用，
# 开机自启注册的也正是这个命令。
uv pip install -e .

Step "生成 config.toml"
if (Test-Path "config.toml") {
    Write-Host "    config.toml 已存在，保持不变" -ForegroundColor DarkGray
} else {
    Copy-Item "config.example.toml" "config.toml"
    Write-Host "    已从 config.example.toml 生成" -ForegroundColor DarkGray
}

Step "环境体检"
& ".\.venv\Scripts\python.exe" -m voicenote --check

Write-Host ""
Write-Host "准备完成。启动方式：" -ForegroundColor Green
Write-Host "    .\.venv\Scripts\pythonw.exe -m voicenote" -ForegroundColor Green
Write-Host ""
Write-Host "想要开机自启：" -ForegroundColor Green
Write-Host "    .\.venv\Scripts\python.exe -m voicenote.autostart enable" -ForegroundColor Green
