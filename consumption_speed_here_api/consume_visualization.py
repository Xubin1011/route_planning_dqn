# Date: 8/2/2023
# Author: Xubin Zhang
# Description: This file contains the implementation of...


import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('../nearest_result_1km.csv')

# 提取需要的数据列
length = df['length']
average_speed = df['average_speed']
average_consumption = df['average_consumption']
# average_consumption_ebus = df['average_consumption_ebus']
# average_consumption_ecar = df['average_consumption_ecar']

# 保存图1为PNG格式
plt.figure(1)
plt.scatter(length/1000, average_speed, color='b')
plt.xlabel('Length(km)')
plt.ylabel('Average Speed(km/h)')
plt.title('Distance traveled and average speed')
plt.savefig('nearest_result_25km_01.png', dpi=300, bbox_inches='tight')

# 保存图2为PDF格式
plt.figure(2)
plt.scatter(length/1000, average_consumption, color='blue', label='Average Consumption')
# plt.scatter(length/1000, average_consumption_ebus, color='blue', label='Average Consumption Ebus')
# plt.scatter(length/1000, average_consumption_ecar, color='red', label='Average Consumption Ecar')
plt.xlabel('Length(km)')
plt.ylabel('Average Consumption(kWh/100km)')
plt.title('Distance traveled and Average Consumption')
plt.legend()
plt.savefig('nearest_result_25km_02.png', bbox_inches='tight')

# 保存图3为PNG格式
plt.figure(3)
plt.scatter(average_speed, average_consumption, color='yellow', label='Average Consumption')
# plt.scatter(average_speed, average_consumption_ebus, color='blue', label='Average Consumption Ebus')
# plt.scatter(average_speed, average_consumption_ecar, color='red', label='Average Consumption Ecar')
plt.xlabel('Average Speed(km/h)')
plt.ylabel('Average Consumption(kWh/100km)')
plt.title('Average Speed and Average Consumption')
plt.legend()
plt.savefig('nearest_result_25km_03.png', dpi=300, bbox_inches='tight')

# 显示绘图
plt.show()
