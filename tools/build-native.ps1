# Builds op2maplib for 32-bit and 64-bit Windows (Release), runs the C++
# tests for both, and copies the DLLs into bindings\dotnet\native\ for the
# .NET binding.
#
# Prerequisites:
#   - Visual Studio 2022 (or its Build Tools) with the "Desktop development
#     with C++" workload, which includes the MSVC x64/x86 compilers and CMake.
#   - cmake on PATH, or the CMake bundled with VS 2022 Build Tools at its
#     default install location (used as a fallback, see below).
#
# Usage (from anywhere; paths are relative to the repository root):
#   powershell -ExecutionPolicy Bypass -File tools\build-native.ps1
#
# Output:
#   build\bin\Release\         64-bit op2maplib.dll and op2maplib_tests.exe
#   build-x86\bin\Release\     32-bit op2maplib.dll and op2maplib_tests.exe
#   bindings\dotnet\native\x64\op2maplib.dll, bindings\dotnet\native\x86\op2maplib.dll
# The Python binding finds the 64-bit DLL in build\bin\Release on its own.
# Stops with an error at the first failed configure, build or test run.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot   # this script lives in <root>\tools

$cmake = (Get-Command cmake -ErrorAction SilentlyContinue).Source
if (-not $cmake) {
    # Not on PATH: try the CMake that ships with VS 2022 Build Tools.
    $cmake = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
}
if (-not (Test-Path $cmake)) { throw "cmake not found" }

# One build tree per architecture. Name = folder under bindings\dotnet\native,
# A = CMake's -A platform for the Visual Studio generator, Dir = build tree.
foreach ($arch in @(@{ Name = "x64"; A = "x64"; Dir = "build" }, @{ Name = "x86"; A = "Win32"; Dir = "build-x86" })) {
    $build = Join-Path $root $arch.Dir
    & $cmake -S $root -B $build -A $arch.A | Out-Null
    if ($LASTEXITCODE) { throw "cmake configure failed ($($arch.Name))" }
    & $cmake --build $build --config Release
    if ($LASTEXITCODE) { throw "build failed ($($arch.Name))" }

    # Run the C++ tests against the profile; show only their last line (the summary).
    $bin = Join-Path $build "bin\Release"
    & (Join-Path $bin "op2maplib_tests.exe") (Join-Path $root "profiles\op2.json") | Select-Object -Last 1
    if ($LASTEXITCODE) { throw "tests failed ($($arch.Name))" }

    $dest = Join-Path $root "bindings\dotnet\native\$($arch.Name)"
    New-Item -ItemType Directory -Force $dest | Out-Null
    Copy-Item (Join-Path $bin "op2maplib.dll") $dest -Force
    Write-Host "copied $($arch.Name) DLL to $dest"
}
