@echo off
chcp 65001 >nul
title JungleAi v2.0

echo ============================
echo   JungleAi v2.0
echo ============================
echo.

:: Проверка Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Python не найден!
    echo Установи Python 3.12+ с python.org
    pause
    exit
)

:: Установка библиотек
echo Установка библиотек...
pip install llama-cpp-python requests --quiet 2>nul

echo.
echo Запуск JungleAi...
python Bot.py
pause
