echo off
cd %~dp0

del /q /f nntl.opensdf nntl.sdf
rd /s /q Debug
rd /s /q Release
rd /s /q ipch
rem rd /s /q x64

del /q /f _extern\gtest-1.7.0\msvc\*.sdf
rd /s /q _extern\gtest-1.7.0\msvc\gtest
rd /s /q _extern\gtest-1.7.0\msvc\x64

rd /s /q examples\Debug
rd /s /q examples\Release
rd /s /q examples\x64

rd /s /q tests\Debug
rd /s /q tests\Release
rd /s /q tests\x64
