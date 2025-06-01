@echo off
echo Activating Python 3.9 environment...
call .\venv39\Scripts\activate.bat

echo Starting Bengali Character Recognition GUI...
python bengali_gui.py

echo.
echo Press any key to exit...
pause > nul
