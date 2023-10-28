#!/bin/bash

try_number=120

interpreter="/home/utlck/.conda/envs/rp/bin/python"
script="/home/utlck/PycharmProjects/route-planning/dqn_noloops.py"
log_name="output_${try_number}.txt"
$interpreter $script $try_number > /home/utlck/PycharmProjects/Tunning_results/"$log_name" 2>&1 &

python_pid=$!
start_time=$(date +%s)
echo -e "running dqn_noloops.py"
echo -e "running dqn_noloops.py"

while true; do
    current_time=$(date +%s)
    elapsed_time=$((current_time - start_time))
    echo -e "\033[A\033[KElapsed time：$((elapsed_time / 60)) mins"
    if ! ps -p $python_pid > /dev/null; then
        echo -e "done"
        break
    fi
    
    sleep 60
done

read -p "Press [Enter] to exit.........."
