#!/bin/bash

interpreter="/home/utlck/.conda/envs/rp/bin/python"
script="/home/utlck/PycharmProjects/route-planning/deployment.py"
log_prefix="deploy_109_500epis_rondom_100"

for ((i=1; i<=100; i++)); do
    log_name="${log_prefix}_$i.txt"
    $interpreter $script >> "/home/utlck/PycharmProjects/Tunning_results/$log_name" 2>&1 &

    python_pid=$!
    start_time=$(date +%s)
    echo -e "Running deploy.py - Iteration $i"

    while true; do
        current_time=$(date +%s)
        elapsed_time=$((current_time - start_time))
        echo -e "\033[A\033[KElapsed time: $((elapsed_time / 1)) s"
        if ! ps -p $python_pid > /dev/null; then
            echo -e "Iteration $i done"
            break
        fi

        sleep 1
    done
done

read -p "Press [Enter] to exit.........."
