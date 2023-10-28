#!/bin/bash

#try_number=057

interpreter="/home/utlck/.conda/envs/rp/bin/python"
script="/home/utlck/PycharmProjects/route-planning/deployment_cs.py"
#log_name="deploy_${try_number}.txt"
log_name="deploy_120_500epis_01_cs.txt"
$interpreter $script > /home/utlck/PycharmProjects/Dij_results/"$log_name" 2>&1 &

python_pid=$!
start_time=$(date +%s)
echo -e "running deploy.py"
echo -e "running deploy.py"

while true; do
    current_time=$(date +%s)
    elapsed_time=$((current_time - start_time))
    echo -e "\033[A\033[KElapsed time：$((elapsed_time / 1)) s"
    if ! ps -p $python_pid > /dev/null; then
        echo -e "done"
        break
    fi
    
    sleep 1
done

read -p "Press [Enter] to exit.........."
