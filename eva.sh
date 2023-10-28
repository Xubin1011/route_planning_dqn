#!/bin/bash

# 初始化变量来保存累加的值
total_success_times=0
total_num_max_q=0
total_total_num_select=0

# 遍历文件
for ((i=1; i<=100; i++)); do
    file_name="/home/utlck/PycharmProjects/Tunning_results/deploy_109_500epis_rondom_100_$i.txt"

    # 使用awk从文件中提取所需的值并累加
    success_times=$(tail -n 4 "$file_name" | awk -F':' '/sucesse times/{print $2}')
    num_max_q=$(tail -n 4 "$file_name" | awk -F':' '/num_max_q/{print $2}')
    total_num_select=$(tail -n 4 "$file_name" | awk -F':' '/total_num_select/{print $2}')


    # 累加到总和
    total_success_times=$((total_success_times + success_times))
    total_num_max_q=$((total_num_max_q + num_max_q))
    total_total_num_select=$((total_total_num_select + total_num_select))
done

# 输出总和
echo "Total Success Times: $total_success_times"
echo "Total Num Max Q: $total_num_max_q"
echo "Total Total Num Select: $total_total_num_select"
