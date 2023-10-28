# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def main():
    # 设置全局变量
    global m, g, rho, A_front, C_d, velocity, sin_alpha, cos_alpha, a, c2

    # 赋值
    m = 1000  # 车辆质量 (kg)
    g = 9.81  # 重力加速度 (m/s^2)
    rho = 1.225  # 空气密度 (kg/m^3)
    A_front = 2.5  # 车辆的前面积 (m^2)
    C_d = 0.3  # 车辆的阻力系数
    velocity = 20  # 车辆的速度 (m/s)
    sin_alpha = 0.1  # sin(alpha) 的值
    cos_alpha = 0.9  # cos(alpha) 的值
    a = 1  # 车辆的加速度 (m/s^2)
    c2 = 100  # 终点的海拔高度 (m)

    # 调用函数计算 P_m
    result = calculate_power_consumption(m, g, rho, A_front, C_d, velocity, sin_alpha, cos_alpha, a, c2)

    # 打印结果
    print(result)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
