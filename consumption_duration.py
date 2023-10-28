# Date: 7/25/2023
# Author: Xubin Zhang
# Description: Calculate consumption between two POIs （in kWh）, get duration between two POIs （in s）

import numpy as np
import random


def haversine(x1, y1, x2, y2):
    # Convert latitude and longitude from degrees to radians
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = np.radians([x1, y1, x2, y2])

    # Haversine formula
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    a = np.sin(delta_lat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    # Earth's mean radius in kilometers
    radius = 6371.0
    distance = radius * c

    # Convert the distance to meters
    distance_meters = distance * 1000

    return distance_meters


#Description: alpha is the slope

def calculate_alpha(x1, y1, c1, x2, y2, c2):
    # Calculate the haversine distance
    distance_meters = haversine(x1, y1, x2, y2)

    # print("Haversine Distance:", distance_meters, "m")
    # Calculate sinalpha based on c2-c1
    elevation_difference = c2 - c1
    if distance_meters != 0:
        sin_alpha = elevation_difference / distance_meters
    else:
        sin_alpha = 0
    cos_alpha = np.sqrt(1 - sin_alpha**2)

    return sin_alpha, cos_alpha, distance_meters

#Description: Calculate consumption (the power needed for vehicle motion) between two POIs
# P_m  = v ( mgsinα + mgC_r cosα +  1/2  ρv^2 A_front C_d  + ma ) (in W)
# m : Mass of the vehicle (in kg)
# g :  Acceleration of gravity (in m/s^2)
# c_r : Coefficient of rolling resistance
# rho : Air density (in kg/m^3)
# A_front : Frontal area of the vehicle (in m^2)
# c_d :  Coefficient of drag
# a :  Acceleration (in m/s^2)
# eta_m: the energy efficiency of transmission, motor and power conversion
# eta_battery: the efficiency of transmission, generator and in-vehicle charger


def consumption_duration(x1, y1, c1, x2, y2, c2, m, g, c_r, rho, A_front, c_d, a, eta_m, eta_battery):

    sin_alpha, cos_alpha, distance_meters = calculate_alpha(x1, y1, c1, x2, y2, c2)
    # random_speed = random.randint(80, 100) # in km/h
    random_speed = random.randint(60, 80)  # in km/h
    average_speed = random_speed * 1000 /3600 #in m/s
    typical_duration = distance_meters / average_speed # in s

    mgsin_alpha = m * g * sin_alpha
    mgCr_cos_alpha = m * g * c_r * cos_alpha
    air_resistance = 0.5 * rho * (average_speed ** 2) * A_front * c_d
    ma = m * a

    power = average_speed * (mgsin_alpha + mgCr_cos_alpha + air_resistance + ma) / eta_m

    # Recuperated energy
    if power < 0:
        if average_speed < 4.17: # 4.17m/s = 15km/h
            power = 0
        else:
            power = power * eta_battery
            if power < -150000:  # 150kW
                power = -150000



    consumption = power * typical_duration / 3600 / 1000 * 1.5  #(in kWh)

    return consumption, typical_duration, distance_meters #(in kWh, s, m)

# # test eCitaro 2 Türen
# x1, y1, c1 = 52.66181,13.38251, 47
# x2, y2, c2 = 51.772324,12.402652,88
# m = 13500 #(Leergewicht)
# g = 9.81
# rho = 1.225
# A_front = 10.03
# c_r = 0.01
# c_d = 0.7
# a = 0
# eta_m, eta_battery = 0.8, 0.8
# consumption, typical_duration, length_meters = consumption_duration(x1, y1, c1, x2, y2, c2, m, g, c_r, rho, A_front, c_d, a, eta_m, eta_battery)
# print("Typical Duration:", typical_duration, "s")
# print("Consumption:", consumption, "kWh")
# print("Average comsuption:", consumption/length_meters*100000, "kWh/100km")


# x1, y1, c1 = 49.403861, 9.390352, 228
#
# x2, y2, c2 = 51.557302, 12.9661, 86
# sin_alpha, cos_alpha=calculate_alpha(x1, y1, c1, x2, y2, c2)
# print(sin_alpha)
# print(cos_alpha)

