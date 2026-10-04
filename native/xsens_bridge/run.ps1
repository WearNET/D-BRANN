param(
    [string]$Target = "xsens_stream_bridge"
)

$ErrorActionPreference = "Stop"

$XSENS_LIB = "C:\Program Files\Xsens\MT Software Suite 2022.2\MT SDK\x64\lib"
$EXE_FILE = ".\build\$Target.exe"

if (-not (Test-Path $EXE_FILE))
{
    Write-Host "Executable not found: $EXE_FILE"
    Write-Host "Run: .\build.ps1 $Target"
    exit 1
}

$env:PATH = "$XSENS_LIB;$env:PATH"

& $EXE_FILE
