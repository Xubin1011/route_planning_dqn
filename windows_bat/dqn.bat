@echo off
setlocal enabledelayedexpansion

set interpreter="E:\Program Files\miniconda3\envs\rp\python.exe"
set script="G:\OneDrive\Thesis\Code\route-planning\dqn_n_actions.py"
set folder_path=G:\Tuning results
set try_number=040

echo Running dqn_n_actions_%try_number%.py...

for /f "tokens=1-3 delims=:" %%a in ("%TIME%") do (
    set "start_hour=%%a"
    set "start_minute=%%b"
    set "start_second=%%c"
)

%interpreter% %script% %try_number% >> "%folder_path%\output_%try_number%.txt"

for /f "tokens=1-3 delims=:" %%a in ("%TIME%") do (
    set "end_hour=%%a"
    set "end_minute=%%b"
    set "end_second=%%c"
)

if !end_hour! lss !start_hour! (
    set /a "hours=(24 + end_hour) - start_hour"
) else (
    set /a "hours=end_hour-start_hour"
)

if !end_minute! lss !start_minute! (
    set /a "hours=hours-1"
    set /a "minutes=(60 + end_minute) - start_minute"
) else (
    set /a "minutes=end_minute-start_minute"
)

if !end_second! lss !start_second! (
    set /a "minutes=minutes-1"
    set /a "seconds=(60 + end_second) - start_second"
) else (
    set /a "seconds=end_second-start_second"
)

echo Script completed in !hours!:!minutes!:!seconds!

for /l %%i in (1,1,2) do (
    powershell -c "(New-Object Media.SoundPlayer 'C:\Windows\Media\Alarm02.wav').PlaySync();"
)

pause
