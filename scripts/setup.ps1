param([ValidateSet('setup','models','diagnose')][string]$Mode = 'setup')
$ErrorActionPreference = 'Stop'
$voiceRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $voiceRoot
$env:VOICEAI_ROOT = $voiceRoot
$env:UV_CACHE_DIR = Join-Path $voiceRoot 'cache\uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $voiceRoot 'python'
$env:UV_PYTHON_BIN_DIR = Join-Path $voiceRoot 'tools\python-bin'
$env:HF_HOME = Join-Path $voiceRoot 'models\huggingface'
$env:TORCH_HOME = Join-Path $voiceRoot 'models\torch'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:GRADIO_ANALYTICS_ENABLED = 'False'
$env:PYTHONUTF8 = '1'
$env:PATH = (Join-Path $voiceRoot 'tools') + ';' + $env:PATH
$env:TEMP = Join-Path $voiceRoot 'cache\tmp'
$env:TMP = $env:TEMP
foreach ($folder in @('tools','models','logs','cache\tmp','vendor')) {
    New-Item -ItemType Directory -Force -Path (Join-Path $voiceRoot $folder) | Out-Null
}

function Run-Checked([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed (exit $LASTEXITCODE): $Program" }
}

# Windows PowerShell 5.1 strips embedded double quotes in native -c arguments.
# Write Python source to a UTF-8 file and pass only its path to the interpreter.
function Run-PythonCode([string]$PythonExe, [string]$Code) {
    $scriptPath = Join-Path $env:TEMP ('voiceai-' + [guid]::NewGuid().ToString('N') + '.py')
    try {
        [System.IO.File]::WriteAllText($scriptPath, $Code, [System.Text.UTF8Encoding]::new($false))
        Run-Checked $PythonExe @($scriptPath)
    } finally {
        Remove-Item -LiteralPath $scriptPath -ErrorAction SilentlyContinue
    }
}

$uiPython = Join-Path $voiceRoot '.venv-ui\Scripts\python.exe'
$ttsPython = Join-Path $voiceRoot '.venv-tts\Scripts\python.exe'
$vcPython = Join-Path $voiceRoot '.venv-vc\Scripts\python.exe'
Start-Transcript -Path (Join-Path $voiceRoot "logs\$Mode.log") -Append | Out-Null
try {
    if ($Mode -eq 'diagnose') {
        Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,OSArchitecture | Format-List
        Get-CimInstance Win32_Processor | Select-Object Name | Format-List
        Get-CimInstance Win32_ComputerSystem | Select-Object @{n='RAM_GiB';e={[math]::Round($_.TotalPhysicalMemory/1GB,1)}} | Format-List
        if (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue) {
            & nvidia-smi.exe --query-gpu=name,memory.total,driver_version --format=csv
        }
        foreach ($pythonExe in @($ttsPython,$vcPython)) {
            if (Test-Path -LiteralPath $pythonExe) {
                Run-PythonCode $pythonExe 'import torch; print("torch",torch.__version__,"CUDA",torch.version.cuda,"available",torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU unavailable")'
            } else { Write-Host "Not installed: $pythonExe" }
        }
    } elseif ($Mode -eq 'models') {
        if (-not (Test-Path -LiteralPath $ttsPython) -or -not (Test-Path -LiteralPath $vcPython)) {
            throw 'Run 01_setup.bat first.'
        }
        Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue
        Remove-Item Env:TRANSFORMERS_OFFLINE -ErrorAction SilentlyContinue
        Write-Host 'Downloading free model weights and checking GPU loading. This may take a while.'
        Run-Checked $ttsPython @('workers\inference.py','tts','--prepare')
        Run-Checked $vcPython @('workers\inference.py','vc','--prepare')
        Run-Checked $vcPython @('workers\inference.py','asr','--prepare')
        Write-Host 'Models are ready. Run 03_start.bat.'
    } else {
        if (-not [Environment]::Is64BitOperatingSystem) { throw '64-bit Windows is required.' }
        Remove-Item -LiteralPath (Join-Path $voiceRoot 'tools\setup-complete.flag') -ErrorAction SilentlyContinue
        $uvExe = Join-Path $voiceRoot 'tools\uv.exe'
        if (-not (Test-Path -LiteralPath $uvExe)) {
            $uvZip = Join-Path $voiceRoot 'cache\uv.zip'
            Invoke-WebRequest -UseBasicParsing -Uri 'https://github.com/astral-sh/uv/releases/download/0.9.7/uv-x86_64-pc-windows-msvc.zip' -OutFile $uvZip
            $actualHash = (Get-FileHash -LiteralPath $uvZip -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($actualHash -ne '5d250c32d3604e28dbe18dc65c668ff628c53e00dde2c642576e831e4a60da64') {
                throw 'uv download checksum mismatch. Stop and retry the download.'
            }
            Expand-Archive -LiteralPath $uvZip -DestinationPath (Join-Path $voiceRoot 'tools') -Force
        }
        Run-Checked $uvExe @('python','install','3.12','3.10')
        foreach ($entry in @(@('.venv-ui','3.12'),@('.venv-tts','3.12'),@('.venv-vc','3.10'))) {
            if (-not (Test-Path -LiteralPath (Join-Path $voiceRoot ($entry[0]+'\Scripts\python.exe')))) {
                Run-Checked $uvExe @('venv','--python',$entry[1],$entry[0])
            }
        }
        Run-Checked $uvExe @('pip','install','--python',$uiPython,'-r','locks\windows-ui.txt')
        Run-PythonCode $uiPython 'import imageio_ffmpeg,shutil; shutil.copyfile(imageio_ffmpeg.get_ffmpeg_exe(),"tools/ffmpeg.exe")'
        Run-Checked $uvExe @('pip','install','--python',$ttsPython,'torch==2.6.0','torchaudio==2.6.0','--index-url','https://download.pytorch.org/whl/cu124')
        Run-Checked $uvExe @('pip','install','--python',$ttsPython,'-r','locks\windows-tts.txt')
        Run-Checked $uvExe @('pip','install','--python',$vcPython,'torch==2.5.1','torchaudio==2.5.1','--index-url','https://download.pytorch.org/whl/cu124')
        Run-Checked $uvExe @('pip','install','--python',$vcPython,'-r','locks\windows-vc.txt')
        $seedDir = Join-Path $voiceRoot 'vendor\seed-vc'
        $revision = '51383efd921027683c89e5348211d93ff12ac2a8'
        if (-not (Test-Path -LiteralPath (Join-Path $seedDir 'inference.py'))) {
            $seedZip = Join-Path $voiceRoot 'cache\seed-vc.zip'
            Invoke-WebRequest -UseBasicParsing -Uri "https://codeload.github.com/Plachtaa/seed-vc/zip/$revision" -OutFile $seedZip
            Expand-Archive -LiteralPath $seedZip -DestinationPath (Join-Path $voiceRoot 'vendor') -Force
            Move-Item -LiteralPath (Join-Path $voiceRoot "vendor\seed-vc-$revision") -Destination $seedDir
        }
        foreach ($pythonExe in @($uiPython,$ttsPython,$vcPython)) {
            Run-Checked $uvExe @('pip','check','--python',$pythonExe)
        }
        Run-PythonCode $ttsPython 'import torch,qwen_tts; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))'
        Run-PythonCode $vcPython 'import torch,librosa,dac; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))'
        & $uvExe pip freeze --python $ttsPython | Out-File -Encoding utf8 'logs\installed-tts.txt'
        & $uvExe pip freeze --python $vcPython | Out-File -Encoding utf8 'logs\installed-vc.txt'
        Set-Content -LiteralPath (Join-Path $voiceRoot 'tools\setup-complete.flag') -Value '0.1.0' -Encoding ascii
        Write-Host 'Setup completed. Next: 02_prepare_models.bat (downloads model weights).'
    }
} catch {
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Log: $voiceRoot\logs\$Mode.log"
    exit 1
} finally { Stop-Transcript | Out-Null }
