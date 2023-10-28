#!/bin/bash
start_time=$(date +%s)
try_number=044

interpreter="/home/utlck/.conda/envs/rp/bin/python"
script="/home/utlck/PycharmProjects/route-planning/dqn_noloops.py"
log_name="output_${try_number}.txt"
log_file="/home/utlck/PycharmProjects/Tunning_results/$log_name"
$interpreter $script $try_number 2>&1 | tee "$log_file"

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))

echo "Total elapsed time: $((elapsed_time / 60)) mins" | tee -a "$log_file"

read -p "Press [Enter] to exit.........."
