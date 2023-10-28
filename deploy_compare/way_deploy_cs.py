# Date: 2023-08-16
# Author: Xubin Zhang
# Description: Calculate the needed information between two pois
# Input: current location, current node, next node
# Output: Distance between next node and target, time and energy consumption between two points, location of next node, power if next node is a CS
# from nearest_location import nearest_location
# from consumption_duration import consumption_duration
# from consumption_duration import haversine
import pandas as pd
import heapq
import numpy as np
import random
from global_var_dij import initial_data_p, initial_data_ch, data_p, data_ch, file_path_p, file_path_ch

def reset_df():
    global data_ch, data_p
    # print(len(data_ch), len(data_p))
    data_ch = initial_data_ch.copy()
    data_p = initial_data_p.copy()
    # print(len(data_ch), len(data_p))


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

def consumption_duration(x1, y1, c1, x2, y2, c2, m, g, c_r, rho, A_front, c_d, a, eta_m, eta_battery):

    # typical_duration, length_meters, average_speed = get_typical_route_here(x1, y1, x2, y2)  # s, m, m/s

    sin_alpha, cos_alpha, distance_meters = calculate_alpha(x1, y1, c1, x2, y2, c2)
    # random_speed = random.randint(80, 100) # in km/h
    random_speed = random.randint(60, 80)  # in km/h
    # random_speed = 80
    average_speed = random_speed * 1000 /3600 #in m/s
    typical_duration = distance_meters / average_speed # in s
    # print(average_speed)
    # print(typical_duration)
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
    consumption = power * typical_duration / 3600 / 1000 *1.5  #(in kWh)
    return consumption, typical_duration, distance_meters #(in kWh, s, m)

class way():
    def __init__(self):
        # initialization
        self.n_ch = 6  # Number of the nearest charging stations
        self.n_p = 4  # Number of the nearest parking lots
        self.n_pois = 10

        self.x_source = 49.0130  # source
        self.y_source = 8.4093
        # self.c_source = 47
        self.x_target = 52.5253  # target
        self.y_target = 13.3694
        # self.c_target = 88
        self.m = 13500  # (Leergewicht) in kg
        self.g = 9.81
        self.rho = 1.225
        self.A_front = 10.03
        self.c_r = 0.01
        self.c_d = 0.7
        self.a = 0
        self.eta_m = 0.82
        self.eta_battery = 0.82

    # update coordinates of target, select the closest CH or P as target
        min_dis = None
        for index, row in initial_data_ch.iterrows():
            distance = haversine(row['Latitude'], row['Longitude'], self.x_target, self.y_target)
            if min_dis is None or distance < min_dis:
                min_dis = distance
                self.closest_index_ch = index
        closest_point_ch = initial_data_ch.loc[self.closest_index_ch]
        self.x_target_ch = closest_point_ch['Latitude']
        self.y_target_ch = closest_point_ch['Longitude']
        print("target_ch:", self.x_target_ch, self.y_target_ch, self.closest_index_ch)
        min_dis = None
        for index, row in initial_data_p.iterrows():
            distance = haversine(row['Latitude'], row['Longitude'], self.x_target, self.y_target)
            if min_dis is None or distance < min_dis:
                min_dis = distance
                self.closest_index_p = index
        closest_point_p = initial_data_p.loc[self.closest_index_p]
        self.x_target_p = closest_point_p['Latitude']
        self.y_target_p = closest_point_p['Longitude']
        print("target_p :", self.y_target_p, self.y_target_p, self.closest_index_p)



    def geo_coord(self, node, index):
        if node in range(self.n_ch):
            Latitude, Longitude, Elevation, Power = initial_data_ch.iloc[int(index)]
            return Latitude, Longitude, Elevation, Power
        else:
            Latitude, Longitude, Altitude = initial_data_p.iloc[int(index)]
            power = None
            return Latitude, Longitude, Altitude, power

    def nearest_location(self, path, x1, y1, n):
        if path == file_path_ch:
            data = data_ch
            # print(len(data_ch))
        else:
            data = data_p
            # print(len(data_p))
        latitudes = data["Latitude"]
        longitudes = data["Longitude"]

        # Priority queue to store information of the n nearest locations
        closest_locations = []

        # Iterate through all the locations
        for lat, lon in zip(latitudes, longitudes):
            # Calculate the distance
            distance = haversine(x1, y1, lat, lon)
            if distance < 25000: # min. driving distance 25km
            # if distance < 50000: # min. driving distance 50km
                continue

            dis_current = haversine(x1, y1, self.x_target_ch, self.y_target_ch)
            dis_next = haversine(lat, lon, self.x_target_ch, self.y_target_ch)
            if dis_current <= dis_next: # only select pois close to the target
                continue

            # negate the distance to find the farthest distance,
            # closest_locations[0] is the farthest location now
            neg_distance = -distance

            # If the number of locations in the queue is less than n,
            # insert the current location
            if len(closest_locations) < n:
                heapq.heappush(closest_locations, (neg_distance, lat, lon))
            else:
                # find the farthest location in the current queue
                min_neg_distance, _, _ = closest_locations[0]

                # If the current location is closer, replace the farthest location
                if neg_distance > min_neg_distance:
                    heapq.heappop(closest_locations)  # pop the farthest location
                    heapq.heappush(closest_locations, (neg_distance, lat, lon))  # insert the closer location

        # convert the distance back to positive values
        closest_locations = pd.DataFrame(closest_locations, columns=["Neg_Distance", "Latitude", "Longitude"])
        closest_locations["Distance"] = -closest_locations["Neg_Distance"]
        closest_locations.drop(columns=["Neg_Distance"], inplace=True)

        # Sort by distance in ascending order
        closest_locations.sort_values(by="Distance", inplace=True)

        # Extract information of the n nearest locations
        nearest_locations = closest_locations.head(n).reset_index(drop=True)
        return nearest_locations

    def info_way(self, node_current, x_current, y_current, alti_current, node_next):
        global data_ch, data_p

        # Obtain n_ch nearest charging stations and n_p nearest parking lots, saving in list nearest_n
        nearest_n = []
        poi_files = [file_path_ch, file_path_p]
        n_values = [self.n_ch, self.n_p]
        for poi_file, n in zip(poi_files, n_values):
            nearest_poi = self.nearest_location(poi_file, x_current, y_current, n)
            nearest_n_tem = nearest_poi.values.tolist()
            if len(nearest_n_tem) < n:

                # nearest_n.append(nearest_n_tem[-1])

                for _ in range(len(nearest_n_tem), n):
                    if poi_file == file_path_ch:
                        nearest_n.append([self.x_target_ch, self.y_target_ch, 0])
                    else:
                        nearest_n.append([self.x_target_p, self.y_target_p, 0])
            nearest_n.extend(nearest_n_tem)
        # print("nearest n locations:", nearest_n)

        # # Map the coordinates to the next location
        # coordinates_dict = {}
        # for i in range(self.n_pois):
        #     coordinates_dict[i] = nearest_n[i]
        # # print(coordinates_dict)
        #
        # # Calculate the time and energy consumption between two points, the distance between next node and target
        # next_x, next_y = coordinates_dict[node_next]
        next_x, next_y, _ = nearest_n[int(node_next)]
        # print(next_x, next_y)

        # Obtain the index of next poi
        if node_next in range(self.n_ch):
            index_next = initial_data_ch[(initial_data_ch["Latitude"] == next_x) & (initial_data_ch["Longitude"] == next_y)].index.values[0]
        else:
            index_next = initial_data_p[(initial_data_p["Latitude"] == next_x) & (initial_data_p["Longitude"] == next_y)].index.values[0]
        _, _, alti_next, power_next = self.geo_coord(node_next, index_next)

        d_next = haversine(next_x, next_y, self.x_target, self.y_target)

        consumption, typical_duration, length_meters = consumption_duration(x_current, y_current, alti_current,
                                                                            next_x, next_y,
                                                                            alti_next, self.m, self.g,
                                                                            self.c_r, self.rho, self.A_front,
                                                                            self.c_d, self.a, self.eta_m,
                                                                            self.eta_battery)

        # delete current node
        if node_current in range(self.n_ch):
            global data_ch
            #indices of the same point in initial_data_ch or data_ch are different
            if x_current != next_x:
                index_current = data_ch[
                    (data_ch["Latitude"] == x_current) & (data_ch["Longitude"] == y_current) & (data_ch["Elevation"] == alti_current)].index.values[0]
                data_ch = data_ch.drop(index_current)
        else:
            global data_p
            if x_current != next_x:
                index_current = data_p[
                    (data_p["Latitude"] == x_current) & (data_p["Longitude"] == y_current) & (data_p["Altitude"] == alti_current)].index.values[0]
                data_p = data_p.drop(index_current)

            # # delete all selected nearest node
            # for poi_file, n in zip(poi_files, n_values):
            #     # global data_ch, data_p
            #     if poi_file == file_path_ch:
            #         for t in range(0, n):
            #             select_x, select_y, _ = nearest_n[t]
            #             if select_x == self.x_target_ch and select_y == self.y_target_ch:
            #                 continue
            #             else:
            #                 match = data_ch[
            #                     (data_ch['Latitude'] == select_x) & (data_ch['Longitude'] == select_y)]
            #                 if not match.empty:
            #                     data_ch = data_ch.drop(match.index)
            #     else:
            #         for t in range(self.n_ch, self.n_pois):
            #             select_x, select_y, _ = nearest_n[t]
            #             if select_x == self.x_target_p and select_y == self.y_target_p:
            #                 continue
            #             else:
            #                 match = data_p[(data_p['Latitude'] == select_x) & (data_p['Longitude'] == select_y)]
            #                 if not match.empty:
            #                     data_p = data_p.drop(match.index)

        return (index_next, next_x, next_y, d_next, power_next, consumption, typical_duration, length_meters)

# #test
# x1 = 49.403861
# y1 = 9.390352
# node_c = 1
# node_next = 5
# myway = way()
# next_x, next_y, d_next, power_next, consumption, typical_duration, length_meters = myway.info_way(node_c, x1, y1, node_next)
#
# print(next_x, next_y, d_next, power_next, consumption, typical_duration, length_meters)
# print("Length, average speed, average consumption", length_meters / 1000, "km", length_meters / typical_duration * 3.6, "km/h", consumption / length_meters * 100000, "kWh/100km\n")



