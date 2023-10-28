# Date: 2023/9/24
# Author: Xubin Zhang
# Description:
# def dijkstra_pois(): Divide a large bbox into 20*20 small bboxes,
# and select the charging station with the maximum charging power closest to the center in each small bbox.
# At last visualize it.
# def dijkstra_edges: Calculate the weight between two vertices.
# If one of the following conditions is met, the weight of an edge is set to infinity.
# The driving time between two vertices is greater than 4.5 hours.
# The energy consumption between the two vertices is greater than 588kWh.
# The distance between two vertices is less than 25km or greater than max_edge_length.
import pandas as pd
import numpy as np
import folium
import random

#initialization
x_source = 49.0130 #source
y_source = 8.4093
x_target = 52.5253 #target
y_target = 13.3694
mass = 13500 #(Leergewicht) in kg
g = 9.81
rho = 1.225
A_front = 10.03
c_r = 0.01
c_d = 0.7
a = 0
eta_m = 0.82
eta_battery = 0.82

max_edge_length = 50000 # in m
speed = 80 # in km/h
####################################################################
def bounding_box(source_lat, source_lon, target_lat, target_lon):
    # Calculate the North latitude, West longitude, South latitude, and East longitude
    south_lat = min(source_lat, target_lat)
    west_lon = min(source_lon, target_lon)
    north_lat = max(source_lat, target_lat)
    east_lon = max(source_lon, target_lon)
    return south_lat, west_lon, north_lat, east_lon
###################################################################
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
################################################################
#Description: alpha is the slope
def calculate_alpha(x1, y1, c1, x2, y2, c2):
    # Calculate the haversine distance
    distance_meters = haversine(x1, y1, x2, y2)
    # Calculate sinalpha based on c2-c1
    elevation_difference = c2 - c1
    slope = np.arctan(elevation_difference / distance_meters)  # (slope in radians) slope belongs to -pi/2 to pi/2
    sin_alpha = np.sin(slope)
    cos_alpha = np.cos(slope)

    return sin_alpha, cos_alpha, distance_meters
###################################################################
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
    # random_speed = random.randint(60, 80)  # in km/h
    # average_speed = random_speed * 1000 / 3600  # in m/s
    average_speed = speed * 1000 / 3600
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
    consumption = power * typical_duration / 3600 / 1000 *1.5 #(in kWh)
    return consumption, typical_duration, distance_meters #(in kWh, s, m)
###############################################################################

# Obtain the location of charging stations in 100 small bbox
def dijkstra_pois():
    # Read the CSV file
    df = pd.read_csv('../cs_combo_150_bbox.csv')

    # Create an empty list to store the information of charging stations with the maximum power
    dijkstra_pois = []

    south_lat, west_lon, north_lat, east_lon = bounding_box(x_source, y_source, x_target, y_target)
    # Create a map
    m = folium.Map(location=[(south_lat + north_lat) / 2, (west_lon + east_lon) / 2], zoom_start=10)

    # Obtain 100 small bbox
    for i in range(20):
        for j in range(20):
            # Calculate the boundaries of the current small bbox
            bbox_south_lat = south_lat + (north_lat - south_lat) * i / 20
            bbox_north_lat = south_lat + (north_lat - south_lat) * (i + 1) / 20
            bbox_west_lon = west_lon + (east_lon - west_lon) * j / 20
            bbox_east_lon = west_lon + (east_lon - west_lon) * (j + 1) / 20

            # Add the boundaries of the small bbox to the map
            folium.Rectangle(bounds=[(bbox_south_lat, bbox_west_lon), (bbox_north_lat, bbox_east_lon)],
                             color='blue').add_to(m)

            # Filter charging station data within the current small bbox
            bbox_filtered_df = df[(df['Latitude'] >= bbox_south_lat) &
                                  (df['Latitude'] <= bbox_north_lat) &
                                  (df['Longitude'] >= bbox_west_lon) &
                                  (df['Longitude'] <= bbox_east_lon)]

            if not bbox_filtered_df.empty:
                # Find charging stations with the maximum power
                max_power_stations = bbox_filtered_df[bbox_filtered_df['Power'] == bbox_filtered_df['Power'].max()]

                # If there is only one charging station with the maximum power, save its coordinates and power
                if len(max_power_stations) == 1:
                    max_power_station = max_power_stations.iloc[0]
                    dijkstra_pois.append([max_power_station['Latitude'],
                                          max_power_station['Longitude'],
                                          max_power_station['Elevation'],
                                          max_power_station['Power']])
                else:
                    # Calculate the center of the current bbox
                    bbox_center_lat = (bbox_south_lat + bbox_north_lat) / 2
                    bbox_center_lon = (bbox_west_lon + bbox_east_lon) / 2

                    # Find the nearest charging station to the center of the bbox among those with the maximum power
                    nearest_station = None
                    min_distance = float('inf')

                    for _, row in max_power_stations.iterrows():
                        station_lat = row['Latitude']
                        station_lon = row['Longitude']
                        station_distance = haversine(bbox_center_lat, bbox_center_lon, station_lat, station_lon)

                        if station_distance < min_distance:
                            nearest_station = row
                            min_distance = station_distance

                    # Save the coordinates and power of the nearest charging station to dijkstra_pois
                    dijkstra_pois.append([nearest_station['Latitude'],
                                          nearest_station['Longitude'],
                                          nearest_station['Elevation'],
                                          nearest_station['Power']])

    # Add markers for selected charging stations with their coordinates and power to the map
    for lat, lon, elevation, power in dijkstra_pois:
        folium.Marker(location=[lat, lon],
                      popup=f'Latitude: {lat}<br>Longitude: {lon}<br>Elevation: {elevation}<br>Power: {power}',
                      icon=folium.Icon(color='green')).add_to(m)

    # Save the map as HTML file named dijkstra_pois.html
    # m.save('G:\OneDrive\Thesis\Code\Dij_results\dijkstra_pois.html')
    #/ home / utlck / PycharmProjects / Dij_results
    m.save('/home/utlck/PycharmProjects/Dij_results/dijkstra_pois_150.html')

    # Save the selected points to a CSV file named dijkstra_pois.csv
    dijkstra_df = pd.DataFrame(dijkstra_pois, columns=['Latitude', 'Longitude', 'Elevation', 'Power'])
    # dijkstra_df.to_csv('G:\OneDrive\Thesis\Code\Dij_results\dijkstra_pois.csv', index=False)
    dijkstra_df.to_csv('/home/utlck/PycharmProjects/Dij_results/dijkstra_pois_150.csv', index=False)


#################################################################

def dijkstra_edges(max_edge_length):
    # data = pd.read_csv("G:\OneDrive\Thesis\Code\Dij_results\dijkstra_pois.csv")
    data = pd.read_csv("/home/utlck/PycharmProjects/Dij_results/dijkstra_pois_150.csv")

    # Obtain Latitude、Longitude、Elevation、Power
    latitude = data["Latitude"].values
    longitude = data["Longitude"].values
    elevation = data["Elevation"].values
    power = data["Power"].values

    south_lat, west_lon, north_lat, east_lon = bounding_box(x_source, y_source, x_target, y_target)
    # Create a map
    m = folium.Map(location=[(south_lat + north_lat) / 2, (west_lon + east_lon) / 2], zoom_start=10)

    num_stations = len(latitude)
    # Creat a list to save weight
    weight_matrix = np.full((num_stations, num_stations), np.inf)

    t_stay = np.full((num_stations, num_stations), 0)

    # visualize all pois
    for i in range(num_stations):
        label = f"Latitude: {latitude[i]:.2f}, Longitude: {longitude[i]:.2f}, Elevation: {elevation[i]:.2f}, Power: {power[i]:.2f}"
        folium.Marker(location=[latitude[i], longitude[i]],
                      popup=label,
                      icon=folium.Icon(color='green')).add_to(m)

    # obtain the coordinate of the target
    n_latitude = latitude[-1]
    n_longitude = longitude[-1]

    # calculate weights of all edges
    for i in range(num_stations):
        for j in range(num_stations):
            if i != j:
                # calculate the distance from two vertices of an edge to the target
                distance_i_to_n = haversine(latitude[i], longitude[i], n_latitude, n_longitude)
                distance_j_to_n = haversine(latitude[j], longitude[j], n_latitude, n_longitude)
                # For example: an edge with two vertices A and B, if the distance from B to target is shorter, the direction of the edge is from A to B
                if distance_i_to_n > distance_j_to_n: #from i to j
                    consumption, typical_duration, distance_meters = consumption_duration(
                        latitude[i], longitude[i], elevation[i],
                        latitude[j], longitude[j], elevation[j],
                        mass, g, c_r, rho, A_front, c_d, a, eta_m, eta_battery
                    )
                    t_stay[i][j] = consumption / power[j]
                    weight_matrix[i][j] = typical_duration + t_stay[i][j]
                    # weight_matrix[i][j] = typical_duration# unconstrained

                    # if consumption is greater than battery capacity, delete this edge
                    if consumption > 588: # 588kWh
                        weight_matrix[i][j] = np.inf

                    # if driving time is greater than 4.5h, then delete this edge
                    if typical_duration > 4.5 * 3600:
                        weight_matrix[i][j] = np.inf

                    if distance_meters < 25000 or distance_meters > max_edge_length:
                        weight_matrix[i][j] = np.inf

                    # visualize the edge
                    if weight_matrix[i][j] != np.inf:
                        folium.PolyLine([(latitude[i], longitude[i]), (latitude[j], longitude[j])], color='blue', weight=1).add_to(m)

    # save weights
    weight_df = pd.DataFrame(weight_matrix)
    # weight_df.to_csv(f"G:\OneDrive\Thesis\Code\Dij_results\dijkstra_edges_{int(max_edge_length/1000)}_{speed}km_h.csv", index=False, header=True)
    weight_df.to_csv(
        f"/home/utlck/PycharmProjects/Dij_results/dijkstra_edges_{int(max_edge_length / 1000)}_{speed}km_h_1_5_150.csv",
        index=False, header=True)

    # save map
    # m.save(f"G:\OneDrive\Thesis\Code\Dij_results\dijkstra_edges_{int(max_edge_length/1000)}_{speed}km_h.html")
    m.save(f"/home/utlck/PycharmProjects/Dij_results/dijkstra_edges_{int(max_edge_length / 1000)}_{speed}km_h_1_5_150.html")


# dijkstra_pois()
dijkstra_edges(max_edge_length)









































