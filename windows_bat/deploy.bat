@echo off

set interpreter="E:\Program Files\miniconda3\envs\rp\python.exe"
set script="G:\OneDrive\Thesis\Code\route-planning\deployment.py"

echo Running deployment%try_number%.py...

%interpreter% %script%  >> "G:\Tuning results\deploy_weights_043.txt"



pause
