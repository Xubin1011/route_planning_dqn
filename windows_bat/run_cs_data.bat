@echo on

set "python_path=E:\Program Files\miniconda3\envs\rp\python.exe"
set "script_path=F:\OneDrive\Thesis\Code\route-planning\cs_data.py"

set "api_key=5b3ce3597851110001cf624880a184fac65b416298dee8f52e43a0fe"
set "file_path=Ladesaeulenregister-processed.xlsx"
set "rows_num=1000"
set "max_retries=2"

"%python_path%" "%script_path%" "%api_key%" "%file_path%" %rows_num% %max_retries% >> cs_data_log.txt

