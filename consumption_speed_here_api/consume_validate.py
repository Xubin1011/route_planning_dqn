# Date: 8/1/2023
# Author: Xubin Zhang
# Description:Randomly select 50 POIs within a bounding box as the sources, 
# and 50 points as the targets. Then calculate the average speed and average consumption of these 50 sections, 
# and compare with the average consumption obtained from the here API

import requests
import time
from consumption_duration import consumption_duration
# import random
import pandas as pd
from nearest_location import nearest_location

m = 13500 #(Leergewicht)
g = 9.81
rho = 1.225
A_front = 10.03
c_r = 0.01
c_d = 0.7
a = 0
eta_m = 0.82
eta_battery = 0.82


def here_ebus(x1, y1, x2, y2):
    api_key = ''
    url = f'https://router.hereapi.com/v8/routes'
    params = {
        "origin": f"{x1},{y1}",
        "destination": f"{x2},{y2}",
        'return': 'summary,typicalDuration',
        # 'spans': 'dynamicSpeedInfo,length,consumption,speedLimit,length',
        'transportMode': 'privateBus',
        'vehicle[speedCap]': '27',
        'vehicle[grossWeight]': '135000',
        'departureTime': 'any',
        'ev[freeFlowSpeedTable]': '0,0.239,27,0.239,45,0.259,60,0.196,75,0.207,90,0.238,100,0.26,110,0.296,120,0.337,130,0.351,250,0.351',
        # 'ev[trafficSpeedTable]': '0,0.349,27,0.319,45,0.329,60,0.266,75,0.287,90,0.318,100,0.33,110,0.335,120,0.35,130,0.36,250,0.36',
        'ev[ascent]': '9',
        'ev[descent]': '4.3',
        'apikey': api_key
    }

    response = requests.get(url, params=params)

    time.sleep(0.5)

    data = response.json()

    try:
        routes = data["routes"]
        if routes:
            typical_duration = data["routes"][0]["sections"][0]["summary"]["typicalDuration"]
            length_meters = data["routes"][0]["sections"][0]["summary"]["length"]
            # base_duration = data["routes"][0]["sections"][0]["summary"]["baseDuration"]
            consumption = data["routes"][0]["sections"][0]["summary"]["consumption"]
        else:
            print(data)
            typical_duration = data
            return typical_duration, 0, 0, 0, 0

    except IndexError:
        print(data)
        typical_duration = data
        return typical_duration, 0, 0, 0, 0

    except KeyError:
        print(data)
        typical_duration = data
        return typical_duration, 0, 0, 0, 0


    average_speed = (length_meters / typical_duration) * 3.6  # (km/h)
    # average_speed = length_meters / base_duration

    average_consumption = consumption / length_meters * 100000  # kWh/100km

    print("average_speed_ebus:", average_speed, "km/h")
    print("average_consumption_ebus:", average_consumption, "kWh/100km")


    # return typical_duration
    return typical_duration, length_meters, average_speed, consumption, average_consumption  # s, m, km/h, kWh/100km


def here_ecar(x1, y1, x2, y2):
    api_key = ''
    url = f'https://router.hereapi.com/v8/routes'
    params = {
        "origin": f"{x1},{y1}",
        "destination": f"{x2},{y2}",
        'return': 'summary,typicalDuration',
        # 'spans': 'dynamicSpeedInfo,length,consumption,speedLimit,length',
        'transportMode': 'car',
        'vehicle[speedCap]': '27',
        # 'vehicle[grossWeight]': '1350',
        # 'departureTime': 'any',
        'ev[freeFlowSpeedTable]': '0,0.239,27,0.239,45,0.259,60,0.196,75,0.207,90,0.238,100,0.26,110,0.296,120,0.337,130,0.351,250,0.351',
        # 'ev[trafficSpeedTable]': '0,0.349,27,0.319,45,0.329,60,0.266,75,0.287,90,0.318,100,0.33,110,0.335,120,0.35,130,0.36,250,0.36',
        'ev[ascent]': '9',
        'ev[descent]': '4.3',
        'apikey': api_key
    }

    response = requests.get(url, params=params)
    data = response.json()
    time.sleep(0.5)

    try:
        routes = data["routes"]
        if routes:
            typical_duration = data["routes"][0]["sections"][0]["summary"]["typicalDuration"]
            length_meters = data["routes"][0]["sections"][0]["summary"]["length"]
            # base_duration = data["routes"][0]["sections"][0]["summary"]["baseDuration"]
            consumption = data["routes"][0]["sections"][0]["summary"]["consumption"]
        else:
            print(data)
            typical_duration = data
            return typical_duration, 0, 0, 0, 0

    except IndexError:
        print(data)
        typical_duration = data
        return typical_duration, 0, 0, 0, 0

    except KeyError:
        print(data)
        typical_duration = data
        return typical_duration, 0, 0, 0, 0


    average_speed = (length_meters / typical_duration) * 3.6  # (km/h)
    # average_speed = length_meters / base_duration
    average_consumption = consumption / length_meters * 100000  # kWh/100km

    print("average_speed_ecar:", average_speed, "km/h")
    print("average_consumption_ecar:", average_consumption, "kWh/100km")

    # print("Departure time:", data["routes"][0]["sections"][0]["departure"]["time"])
    # print("Summary:", data["routes"][0]["sections"][0]["summary"])
    # print("Average speed:", average_speed, "m/s", "=", average_speed * 3.6, "km/h")

    # return typical_duration
    return typical_duration, length_meters, average_speed, consumption, average_consumption  # s, m, m/s


# Randomly select 50 sectoins from a table, and output table validata_pois.csv
def random_poi(file_path):

    # Load data
    df_a = pd.read_csv(file_path)

    # Randomly select 50 rows for source and 50 rows for target
    source_rows = df_a.sample(n=100, random_state=42)
    target_rows = df_a.drop(source_rows.index).sample(n=100, random_state=42)

    # Combine source and target data
    df_b = pd.concat([source_rows[['Latitude', 'Longitude', 'Elevation']],
                      target_rows[['Latitude', 'Longitude', 'Elevation']]], axis=1)

    # Rename columns
    df_b.columns = ['sou_lat', 'sou_lon', 'sou_alt', 'tar_lat', 'tar_lon', 'tar_alt']

    # df_b = df_b.drop(index=range(50, 99))
    # Save the result into b.csv
    df_b.to_csv('validata_pois.csv', index=False)

    return None

# Computer the result based on  consumption_duration(), here_ebus(), here_ecar(), and store in validata_result.csv
def compare_result():
    df = pd.read_csv('validata_pois.csv')

    lengths,typical_durations, average_speeds, consumptions, average_consumptions = [],[],[],[],[]
    lengths_ebus,typical_durations_ebus, average_speeds_ebus, consumptions_ebus, average_consumptions_ebus = [],[],[],[],[]
    lengths_ecar,typical_durations_ecar, average_speeds_ecar, consumptions_ecar, average_consumptions_ecar = [],[],[],[],[]

    for index, row in df.iterrows():
        x1, y1, c1, x2, y2, c2 = row['sou_lat'], row['sou_lon'], row['sou_alt'], row['tar_lat'], row['tar_lon'], row['tar_alt']
        consumption, typical_duration, length_meters = consumption_duration(x1, y1, c1, x2, y2, c2, m, g, c_r, rho, A_front, c_d, a, eta_m,eta_battery)
        average_speed = (length_meters / typical_duration) *3.6  # km/h
        average_consumption = consumption/length_meters*100000 #kWh/100km

        print("average_speed:",average_speed, "km/h")
        print("average_consumption:",average_consumption, "kWh/100km")


        typical_duration_ebus, length_meters_ebus, average_speed_ebus, consumption_ebus, average_consumption_ebus = here_ebus(x1, y1, x2, y2)
        typical_duration_ecar, length_meters_ecar, average_speed_ecar, consumption_ecar, average_consumption_ecar = here_ecar(x1, y1, x2, y2)

        lengths.append(length_meters)
        typical_durations.append(typical_duration)
        average_speeds.append(average_speed)
        consumptions.append(consumption)
        average_consumptions.append(average_consumption)

        lengths_ebus.append(length_meters_ebus)
        typical_durations_ebus.append(typical_duration_ebus)
        average_speeds_ebus.append(average_speed_ebus)
        consumptions_ebus.append(consumption_ebus)
        average_consumptions_ebus.append(average_consumption_ebus)

        lengths_ecar.append(length_meters_ecar)
        typical_durations_ecar.append(typical_duration_ecar)
        average_speeds_ecar.append(average_speed_ecar)
        consumptions_ecar.append(consumption_ecar)
        average_consumptions_ecar.append(average_consumption_ecar)

    df['length'] = lengths
    df['typical_duration'] = typical_durations
    df['average_speed'] = average_speeds
    df['consumption'] = consumptions
    df['average_consumption'] = average_consumptions

    df['length_ebus'] = lengths_ebus
    df['typical_duration_ebus'] = typical_durations_ebus
    df['average_speed_ebus'] = average_speeds_ebus
    df['consumption_ebus'] = consumptions_ebus
    df['average_consumption_ebus'] = average_consumptions_ebus

    df['length_ecar'] = lengths_ecar
    df['typical_duration_ecar'] = typical_durations_ecar
    df['average_speed_ecar'] = average_speeds_ecar
    df['consumption_ecar'] = consumptions_ecar
    df['average_consumption_ecar'] = average_consumptions_ecar

    df.to_csv('validata_result.csv', index=False, header=True)
    return None





def nearest():
    n=2
    file_path = '../cs_combo_bbox.csv'
    df = pd.read_csv('validata_pois.csv')
    cs_data = pd.read_csv('../cs_combo_bbox.csv')

    lengths, typical_durations, average_speeds, consumptions, average_consumptions = [], [], [], [], []

    for index, row in df.iterrows():
        x1, y1, c1 = row['sou_lat'], row['sou_lon'], row['sou_alt']
        nearest_locations = nearest_location(file_path, x1, y1, n)
        print(nearest_locations)
        x2 = nearest_locations.loc[0, 'Latitude']
        y2 = nearest_locations.loc[0, 'Longitude']
        print("x1,y1, c1=",x1, y1,c1)
        print("x2,y2=", x2, y2)
        matched_row = cs_data[(cs_data['Latitude'] == x2) & (cs_data['Longitude'] == y2)]
        c2 = matched_row['Elevation'].values[0]
        print("c2=", c2)
        consumption, typical_duration, length_meters = consumption_duration(x1, y1, c1, x2, y2, c2, m, g, c_r, rho, A_front,c_d, a, eta_m, eta_battery)
        average_speed = (length_meters / typical_duration) * 3.6  # km/h
        average_consumption = consumption / length_meters * 100000  # kWh/100km

        print (average_consumption)

        lengths.append(length_meters)
        typical_durations.append(typical_duration)
        average_speeds.append(average_speed)
        consumptions.append(consumption)
        average_consumptions.append(average_consumption)

    df['length'] = lengths
    df['typical_duration'] = typical_durations
    df['average_speed'] = average_speeds
    df['consumption'] = consumptions
    df['average_consumption'] = average_consumptions

    df.to_csv('nearest_result_25km.csv', index=False, header=True)
    return None



# file_path = 'cs_combo.csv'
# random_poi(file_path)
# compare_result()

nearest()















