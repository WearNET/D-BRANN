param(
    [string]$Target = "xsens_stream_bridge"
)

$ErrorActionPreference = "Stop"

$VS_DEV_SHELL = "C:\Program Files\Microsoft Visual Studio\18\Insiders\Common7\Tools\Launch-VsDevShell.ps1"

if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue))
{
    Write-Host "MSVC environment not loaded. Initializing x64 developer environment..."

    if (-not (Test-Path $VS_DEV_SHELL))
    {
        throw "Visual Studio Developer Shell not found: $VS_DEV_SHELL"
    }

    & $VS_DEV_SHELL `
        -Arch amd64 `
        -HostArch amd64
}

if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue))
{
    throw "MSVC compiler cl.exe could not be initialized."
}

$XSENS_SDK = "C:\Program Files\Xsens\MT Software Suite 2022.2\MT SDK"
$XSENS_INCLUDE = "$XSENS_SDK\x64\include"
$XSENS_LIB = "$XSENS_SDK\x64\lib"

$SOURCE_FILE = ".\src\$Target\$Target.cpp"
$OUTPUT_EXE = ".\build\$Target.exe"

if (-not (Test-Path $SOURCE_FILE))
{
    throw "Source file not found: $SOURCE_FILE"
}

New-Item -ItemType Directory -Force -Path ".\build" | Out-Null

Write-Host ""
Write-Host "Building target: $Target"
Write-Host "Source: $SOURCE_FILE"
Write-Host ""

cl `
    /EHsc `
    /std:c++17 `
    /Fe:$OUTPUT_EXE `
    /I"$XSENS_INCLUDE" `
    $SOURCE_FILE `
    /link `
    /LIBPATH:"$XSENS_LIB" `
    xsensdeviceapi64.lib `
    xstypes64.lib `
    Ws2_32.lib

if ($LASTEXITCODE -ne 0)
{
    throw "Build failed for target: $Target"
}

Write-Host ""
Write-Host "Build complete: $OUTPUT_EXE"
