@echo off
rem Compiles the given .cpp files on their own (no linking) with MSVC, to check
rem them while other parts of the library are still being written.
rem Usage: tools\compile-check.bat src\mapgen\field.cpp [more.cpp ...]
setlocal
set "VSDEV=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
call "%VSDEV%" -arch=x64 -host_arch=x64 >nul || exit /b 1
cd /d "%~dp0.."
if not exist build\check mkdir build\check
cl /nologo /c /std:c++17 /W4 /WX /permissive- /utf-8 /EHsc /O2 /DOP2MAPLIB_STATIC /Iinclude /Isrc /Fobuild\check\ %*
