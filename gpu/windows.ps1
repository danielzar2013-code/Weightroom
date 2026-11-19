# Weightroom GPU for Windows.
# Installs ComfyUI with the Weightroom bridge, Chatterbox speech, Ollama for chat and a Cloudflare tunnel
# into %LOCALAPPDATA%\WeightroomGPU (no admin rights needed), starts them, and opens the Weightroom app
# connected to this PC's NVIDIA GPU. Run it again any time: it skips what is already installed.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$App = 'https://danielzar2013-code.github.io/Weightroom/'
$AppOrigin = 'https://danielzar2013-code.github.io'
$ComfyTag = 'v0.45.5'
$ChatterboxRev = 'f7d7a16187430abcaf91a3039b9c83aa9960816a'
$Root = Join-Path $env:LOCALAPPDATA 'WeightroomGPU'
$Comfy = Join-Path $Root 'ComfyUI'
$Py = Join-Path $Root 'venv\Scripts\python.exe'
$Logs = Join-Path $Root 'logs'
$Port = 8188
New-Item -ItemType Directory -Force -Path $Root, $Logs | Out-Null

function Say([string]$Text) { Write-Host ''; Write-Host "== $Text" -ForegroundColor Cyan }
function Run([string]$Exe, [string[]]$Arguments) {
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Exe failed (exit code $LASTEXITCODE)." }
}
function Get-File([string]$Url, [string]$Out) { Run 'curl.exe' @('-fL', '--retry', '3', '-o', $Out, $Url) }
function Expand-Into([string]$Zip, [string]$Dest) {
    # GitHub zips hold one top folder; move its contents to $Dest.
    $tmp = Join-Path $Root '_unzip'
    if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
    New-Item -ItemType Directory -Force -Path $tmp | Out-Null
    Run 'tar.exe' @('-xf', $Zip, '-C', $tmp)
    $top = Get-ChildItem -Path $tmp | Select-Object -First 1
    if (Test-Path $Dest) { Remove-Item -Recurse -Force $Dest }
    Move-Item -Path $top.FullName -Destination $Dest
    Remove-Item -Recurse -Force $tmp, $Zip
}
function Stop-Old {
    $pidFile = Join-Path $Root 'pids.txt'
    if (Test-Path $pidFile) {
        foreach ($id in Get-Content $pidFile) { Stop-Process -Id ([int]$id) -Force -ErrorAction SilentlyContinue }
        Remove-Item $pidFile -Force
    }
}

try {
    # ---------- GPU ----------
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        throw 'No NVIDIA driver found. Install the latest driver from nvidia.com and run this again.'
    }
    & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    Stop-Old

    # ---------- Python (uv) ----------
    $uv = Join-Path $env:USERPROFILE '.local\bin\uv.exe'
    if (-not (Test-Path $uv)) {
        $found = Get-Command uv -ErrorAction SilentlyContinue
        if ($found) { $uv = $found.Source } else {
            Say 'Installing uv (Python manager)'
            Run 'powershell.exe' @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', 'irm https://astral.sh/uv/install.ps1 | iex')
        }
    }
    if (-not (Test-Path $uv)) { throw 'uv did not install. Close this window and run Weightroom-GPU.bat again.' }

    # ---------- ComfyUI ----------
    if (-not (Test-Path (Join-Path $Comfy 'main.py'))) {
        Say "Downloading ComfyUI $ComfyTag"
        $zip = Join-Path $Root 'comfyui.zip'
        Get-File "https://github.com/Comfy-Org/ComfyUI/archive/refs/tags/$ComfyTag.zip" $zip
        Expand-Into $zip $Comfy
    }
    if (-not (Test-Path $Py)) {
        Say 'Setting up Python 3.12'
        Run $uv @('venv', '--python', '3.12', (Join-Path $Root 'venv'))
    }
    $constraints = Join-Path $Root 'constraints.txt'
    $depsMark = Join-Path $Root '.deps'
    $depsDone = (Test-Path $depsMark) -and ((Get-Content $depsMark -Raw).Trim() -eq $ComfyTag)
    if (-not $depsDone) {
        Say 'Installing PyTorch for NVIDIA (about 3 GB, one time)'
        Run $uv @('pip', 'install', '--python', $Py, 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cu128')
    }
    # Keep the CUDA build of PyTorch: other packages may not replace it with the CPU one.
    $torchVer = (& $Py -c 'import torch; print(torch.__version__)').Trim()
    Set-Content -Path $constraints -Value "torch==$torchVer" -Encoding ASCII
    if (-not $depsDone) {
        Say 'Installing ComfyUI requirements'
        Run $uv @('pip', 'install', '--python', $Py, '-r', (Join-Path $Comfy 'requirements.txt'), 'qrcode', '-c', $constraints)
        Set-Content -Path $depsMark -Value $ComfyTag -Encoding ASCII
    }

    # ---------- add-ons ----------
    $nodes = Join-Path $Comfy 'custom_nodes'
    $bridge = Join-Path $nodes 'weightroom_bridge'
    New-Item -ItemType Directory -Force -Path $bridge | Out-Null
    Get-File ($App + 'gpu/weightroom_bridge/__init__.py') (Join-Path $bridge '__init__.py')
    $cb = Join-Path $nodes 'ComfyUI_Fill-ChatterBox'
    if (-not (Test-Path (Join-Path $cb '.weightroom-ok'))) {
        Say 'Adding Chatterbox speech (Hebrew and 22 more languages)'
        $zip = Join-Path $Root 'chatterbox.zip'
        Get-File "https://github.com/filliptm/ComfyUI_Fill-ChatterBox/archive/$ChatterboxRev.zip" $zip
        Expand-Into $zip $cb
        Run $uv @('pip', 'install', '--python', $Py, '-r', (Join-Path $cb 'requirements.txt'), '-c', $constraints)
        Set-Content -Path (Join-Path $cb '.weightroom-ok') -Value 'ok' -Encoding ASCII
    }

    # ---------- Ollama (chat) ----------
    $ollama = $null
    $found = Get-Command ollama -ErrorAction SilentlyContinue
    if ($found) { $ollama = $found.Source } else {
        $ollama = Join-Path $Root 'ollama\ollama.exe'
        if (-not (Test-Path $ollama)) {
            Say 'Downloading Ollama for chat (about 1.6 GB, one time)'
            $zip = Join-Path $Root 'ollama.zip'
            Get-File 'https://ollama.com/download/ollama-windows-amd64.zip' $zip
            New-Item -ItemType Directory -Force -Path (Join-Path $Root 'ollama') | Out-Null
            Run 'tar.exe' @('-xf', $zip, '-C', (Join-Path $Root 'ollama'))
            Remove-Item -Force $zip
        }
    }

    # ---------- tunnel ----------
    $cf = Join-Path $Root 'cloudflared.exe'
    if (-not (Test-Path $cf)) {
        Get-File 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe' $cf
    }

    # ---------- key ----------
    $keyFile = Join-Path $Root 'key.txt'
    if (-not (Test-Path $keyFile)) {
        $bytes = New-Object byte[] 24
        [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
        $newKey = [Convert]::ToBase64String($bytes).TrimEnd('=').Replace('+', '-').Replace('/', '_')
        Set-Content -Path $keyFile -Value $newKey -Encoding ASCII -NoNewline
    }
    $Key = (Get-Content $keyFile -Raw).Trim()

    # ---------- start ----------
    $env:WEIGHTROOM_KEY = $Key
    $env:WEIGHTROOM_SHARED_GPU = '1'
    $env:OLLAMA_KEEP_ALIVE = '5m'
    $started = @()
    $ollamaUp = $false
    try { Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:11434/api/tags' -TimeoutSec 3 | Out-Null; $ollamaUp = $true } catch { }
    if (-not $ollamaUp -and $ollama -and (Test-Path $ollama)) {
        $started += Start-Process -FilePath $ollama -ArgumentList 'serve' -NoNewWindow -PassThru `
            -RedirectStandardOutput (Join-Path $Logs 'ollama.out.log') -RedirectStandardError (Join-Path $Logs 'ollama.log')
    }

    Say 'Starting ComfyUI (the first start takes a minute or two)'
    $comfyProc = Start-Process -FilePath $Py -WorkingDirectory $Comfy -NoNewWindow -PassThru `
        -ArgumentList @('main.py', '--listen', '127.0.0.1', '--port', "$Port", '--enable-cors-header', $AppOrigin) `
        -RedirectStandardOutput (Join-Path $Logs 'comfyui.log') -RedirectStandardError (Join-Path $Logs 'comfyui.err.log')
    $started += $comfyProc
    ($started | ForEach-Object { $_.Id }) | Set-Content -Path (Join-Path $Root 'pids.txt')
    $ready = $false
    for ($i = 0; $i -lt 300 -and -not $ready; $i++) {
        if ($comfyProc.HasExited) {
            Get-Content (Join-Path $Logs 'comfyui.err.log') -Tail 40
            throw 'ComfyUI stopped. The lines above say why.'
        }
        try {
            Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$Port/weightroom/info" -Headers @{ Authorization = "Bearer $Key" } -TimeoutSec 5 | Out-Null
            $ready = $true
        } catch { Start-Sleep -Seconds 2 }
    }
    if (-not $ready) { throw 'ComfyUI did not start in 10 minutes. See the logs in ' + $Logs }

    Say 'Opening a secure tunnel'
    $tunnelLog = Join-Path $Logs 'tunnel.log'
    if (Test-Path $tunnelLog) { Remove-Item -Force $tunnelLog }
    $started += Start-Process -FilePath $cf -ArgumentList @('tunnel', '--url', "http://127.0.0.1:$Port", '--no-autoupdate') -NoNewWindow -PassThru `
        -RedirectStandardOutput (Join-Path $Logs 'tunnel.out.log') -RedirectStandardError $tunnelLog
    ($started | ForEach-Object { $_.Id }) | Set-Content -Path (Join-Path $Root 'pids.txt')
    $url = $null
    for ($i = 0; $i -lt 60 -and -not $url; $i++) {
        Start-Sleep -Seconds 1
        if (Test-Path $tunnelLog) {
            $m = Select-String -Path $tunnelLog -Pattern 'https://[a-z0-9-]+\.trycloudflare\.com' | Select-Object -First 1
            if ($m) { $url = $m.Matches[0].Value }
        }
    }
    if (-not $url) { throw 'The tunnel did not start. See ' + $tunnelLog }

    $json = '{"u": "' + $url + '", "k": "' + $Key + '"}'
    $blob = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($json)).TrimEnd('=').Replace('+', '-').Replace('/', '_')
    $link = $App + '#gpu=' + $blob

    Say 'Your GPU is ready'
    Write-Host 'The Weightroom app is opening in your browser, connected to this GPU.'
    Write-Host 'On your phone: scan this code, or send yourself the link (it is also copied).'
    Write-Host ''
    Write-Host $link -ForegroundColor Green
    Write-Host ''
    & $Py -c 'import sys, qrcode; q = qrcode.QRCode(border=1); q.add_data(sys.argv[1]); q.print_ascii(invert=True)' $link
    try { Set-Clipboard -Value $link } catch { }
    Start-Process $link
    Write-Host ''
    Write-Host 'Keep this window open while you use your GPU. Close it to stop.' -ForegroundColor Yellow
    Write-Host "Models are saved in $Comfy\models. Logs: $Logs"

    while (-not $comfyProc.HasExited) { Start-Sleep -Seconds 3 }
    Write-Host 'ComfyUI stopped. Last lines of its log:' -ForegroundColor Yellow
    Get-Content (Join-Path $Logs 'comfyui.err.log') -Tail 30
}
catch {
    Write-Host ''
    Write-Host ('Something went wrong: ' + $_.Exception.Message) -ForegroundColor Red
}
finally {
    Stop-Old
    Write-Host ''
    Read-Host 'Press Enter to close'
}
