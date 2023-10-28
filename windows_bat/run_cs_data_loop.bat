@echo on

set loop_count=2

for /l %%i in (1, 1, %loop_count%) do (
    echo Running iteration %%i...
    call run_cs_data.bat
    if %%i NEQ %loop_count% (
        echo Waiting for 10 minutes...
        timeout /t 600 /nobreak
    )
)

echo All iterations completed.

pause

