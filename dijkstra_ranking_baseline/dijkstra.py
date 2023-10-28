#Description: Find the shortest path and check whether the constraints are violated.
# If the constraints are violated, randomly delete a vertex from the shortest path and
# search for the shortest path again until a feasible shortest path is found.

import pandas as pd
import numpy as np
import networkx as nx
import os
import random

from dijkstra_graph import haversine, x_source, y_source, x_target, y_target, consumption_duration

import folium
from data_cleaning.bounding_box import bbox

m = 13500 #(Leergewicht)
g = 9.81
rho = 1.225
A_front = 10.03
c_r = 0.01
c_d = 0.7
a = 0
eta_m, eta_battery = 0.8, 0.8

max_edge_length = 50000 # in m
speed = 80
cs_path = '/home/utlck/PycharmProjects/Dij_results/dijkstra_pois_150.csv'
p_path = '../parking_bbox.csv'
# dij_pois_path = 'G:\OneDrive\Thesis\Code\Dij_results\dijkstra_pois.csv'
# # route_path = f'G:\OneDrive\Thesis\Code\Dij_results\dij_path_{int(max_edge_length/1000)}.csv'
# # weights_path = f'G:\OneDrive\Thesis\Code\Dij_results\dijkstra_edges_{int(max_edge_length/1000)}.csv'
# # map_name = f'G:\OneDrive\Thesis\Code\Dij_results\dij_path_{int(max_edge_length/1000)}.html'
# route_path = f'G:\OneDrive\Thesis\Code\Dij_results\dij_path_{int(max_edge_length/1000)}_6080km_h.csv'
# weights_path = f'G:\OneDrive\Thesis\Code\Dij_results\dijkstra_edges_{int(max_edge_length/1000)}_6080km_h.csv'
# map_name = f'G:\OneDrive\Thesis\Code\Dij_results\dij_path_{int(max_edge_length/1000)}_6080km_h.html'
##########Linux
dij_pois_path = '/home/utlck/PycharmProjects/Dij_results/dijkstra_pois_150.csv'
route_path = f'/home/utlck/PycharmProjects/Dij_results/dij_path_{int(max_edge_length/1000)}_{speed}km_h_1_5_150.csv'
weights_path = f'/home/utlck/PycharmProjects/Dij_results/dijkstra_edges_{int(max_edge_length/1000)}_{speed}km_h_1_5_150.csv'
map_name = f'/home/utlck/PycharmProjects/Dij_results/dij_path_{int(max_edge_length/1000)}_{speed}km_h_1_5_150.html'

stay_list = [0]
distance = [0]
consumption_list = [0]

# select the closest node in graph G
def get_closest_node(G, latitude, longitude):
    closest_node = None
    closest_distance = float('inf')

    for node in G.nodes(data=True):
        node_latitude = node[1]['latitude']
        node_longitude = node[1]['longitude']

        distance = haversine(latitude, longitude, node_latitude, node_longitude)

        if distance < closest_distance:
            closest_node = node[0]
            closest_distance = distance

    return closest_node

# cheack edges
def check_edge(x_current, y_current,ati_current, x_next, y_next, ati_next, power_next):
    global t_stay, t_secd_current, t_secch_current, stay_list, distance
    terminated = False
    consumption, typical_duration, distance_meters = consumption_duration(x_current, y_current, ati_current, x_next, y_next, ati_next, m, g, c_r, rho, A_front, c_d, a, eta_m, eta_battery)
    t_stay = consumption / power_next * 3600 # in s
    stay_list.append(t_stay)
    distance.append(distance_meters)
    consumption_list.append(consumption)
    # the time that arriving next location
    t_arrival = t_secd_current + t_secch_current + typical_duration
    # the depature time when leave next location
    t_departure = t_arrival + t_stay

    ##################################################################
    # check rest, driving time constraint
    if t_arrival >= section:  # A new section begin before arrival next state, only consider the  last section
        t_secd_current = t_arrival % section
        if t_secch_current < min_rest:
            terminated = True
            print("Terminated: Violated max_driving times")
        t_secch_current = t_stay
    else:  # still in current section when arriving next poi
        if t_departure >= section:  # A new section begin before leaving next state,only consider the  last section
            t_secch_current += t_departure % section
            if t_secch_current < min_rest:
                terminated = True
                print("Terminated: Violated max_driving times")
            t_secch_current += (t_stay - t_departure % section)
            t_secd_current = 0
        else:  # still in current section
            t_secch_current = t_stay + t_secch_current
            t_secd_current +=  typical_duration
    return terminated

def check_path(path):
    path_lat = []
    path_lon = []
    path_alt = []
    path_power = []
    global stay_list, distance
    for i in range(len(path)):
        Latitude, Longitude, Elevation, Power = pois_df.iloc[path[i]]
        path_lat.append(Latitude)
        path_lon.append(Longitude)
        path_alt.append(Elevation)
        path_power.append(Power)
        # print(path_lat,path_lon, path_alt, path_power)
    for t in range (len(path) - 1):
        terminated = check_edge(path_lat[t],path_lon[t], path_alt[t], path_lat[t+1], path_lon[t+1], path_alt[t+1], path_power[t+1])
        if terminated:
            unfeasible = True
            stay_list = []
            distance = []
            break
        else:
            unfeasible = False
    return unfeasible

def visualization(cs_path, p_path, route_path, source_lat, source_lon, target_lat, target_lon, map_name):
    # calculate the bounding box
    south_lat, west_lon, north_lat, east_lon = bbox(source_lat, source_lon, target_lat, target_lon)
    # Read data from cs_path
    data1 = pd.read_csv(cs_path)
    # Read data from p_path
    data2 = pd.read_csv(p_path)
    # Calculate the center of the bounding box
    center_lat = (south_lat + north_lat) / 2
    center_lon = (west_lon + east_lon) / 2
    # Create a map object
    map_object = folium.Map(location=[center_lat, center_lon], zoom_start=10)
    # Add the bounding box area to the map
    bbox_coords = [[south_lat, west_lon], [north_lat, west_lon], [north_lat, east_lon], [south_lat, east_lon], [south_lat, west_lon]]
    folium.Polygon(locations=bbox_coords, color='blue', fill=True, fill_color='blue', fill_opacity=0.2).add_to(map_object)
    # Read data from cs_path and add yellow markers for data points
    for _, row in data1.iterrows():
        latitude, longitude = row['Latitude'], row['Longitude']
        folium.CircleMarker(location=[latitude, longitude], radius=1, color='yellow', fill=True, fill_color='yellow').add_to(map_object)
    # # Read data from p_path and add blue markers for data points
    # for _, row in data2.iterrows():
    #     latitude, longitude = row['Latitude'], row['Longitude']
    #     folium.CircleMarker(location=[latitude, longitude], radius=1, color='blue', fill=True, fill_color='blue').add_to(map_object)
    # Read data from route_path as path coordinates
    path_data = pd.read_csv(route_path)
    path_coords = list(zip(path_data['Latitude'], path_data['Longitude']))
    path_infos = list(zip(path_data['Latitude'], path_data['Longitude'], path_data['Stay'], path_data['Distance']))
    for coord in path_infos:
        latitude, longitude, stay, distance = coord
        folium.Marker(location=[latitude, longitude],
                      popup=f'Latitude: {latitude}<br>Longitude: {longitude}<br>Stay: {stay/60}mins<br>Distance: {distance/1000}km',
                      icon=folium.Icon(color='green')).add_to(map_object)
        # folium.CircleMarker(location=[latitude, longitude], radius=2, color='red', fill=True, fill_color='red').add_to(
        #     map_object)
    # Add a red line to represent the path
    folium.PolyLine(locations=path_coords, color='red').add_to(map_object)
    # Save the map as an HTML file and display it
    map_object.save(map_name)

#visualization(cs_path, p_path, route_path, myway.x_source, myway.y_source, myway.x_target, myway.y_target)

def visu(path):
    if os.path.exists(route_path):
        os.remove(route_path)
    path_lat = []
    path_lon = []
    path_power = []
    path_alti = []
    global stay_list, distance, consumption_list
    # print(stay_list)
    for i in range(len(path)):
        Latitude, Longitude, Elevation, Power = pois_df.iloc[path[i]]
        path_lat.append(Latitude)
        path_lon.append(Longitude)
        path_alti.append(Elevation)
        path_power.append(Power)
    geo_coord = pd.DataFrame({'Latitude': path_lat, 'Longitude': path_lon, 'Altitude': path_alti, 'Power': path_power, 'Stay': stay_list, 'Distance': distance, 'Consumption (in kWh)': consumption_list})
    geo_coord.to_csv(route_path, index=False)

    total_distance = sum(distance) / 1000
    print(f"diatance = {total_distance}km")
    totoal_driving = total_distance / speed  # in h
    print(f"driving time = {totoal_driving}h")
    total_cs = sum(stay_list) / 3600 # in h
    print(f"charging time = {total_cs}h")
    total_cs_notarget = (sum(stay_list) - stay_list[-1]) / 3600
    print(f"charging time without target = {total_cs_notarget}h")
    total_consumption = sum(consumption_list)
    print(f"total consumption = {total_consumption} kWh")

    visualization(cs_path, p_path, route_path, x_source, y_source, x_target, y_target, map_name)

    geo_coord['Stay (in min)'] = geo_coord['Stay'] / 60
    geo_coord['Distance (in km)'] = geo_coord['Distance'] / 1000
    geo_coord.to_csv(route_path, index=False)


    output_data = {
        'Metric': ['Total Distance (km)', 'Total Driving Time (h)', 'Total Charging Time (h)',
                   'Charging Time Without Target (h)', 'Total Consumption (kWh)'],
        'Value': [total_distance, totoal_driving, total_cs, total_cs_notarget, total_consumption]
    }
    output_df = pd.DataFrame(output_data)
    merged_df = pd.concat([geo_coord, output_df], ignore_index=True)
    merged_df.to_csv(route_path, index=False)


###########################################


t_stay, t_secd_current, t_secch_current = 0, 0, 0
# Each section has the same fixed travel time
min_rest = 2700  # in s
max_driving = 16200  # in s
section = min_rest + max_driving

# load weights
weights_df = pd.read_csv(weights_path)
weights_matrix = weights_df.values


# load pois
pois_df = pd.read_csv(dij_pois_path)
latitude = pois_df['Latitude'].values
longitude = pois_df['Longitude'].values

# creat graph
G = nx.DiGraph()

# add pois into graph
for i, row in pois_df.iterrows():
    G.add_node(i, latitude=row['Latitude'], longitude=row['Longitude'])

source = get_closest_node(G, x_source, y_source)
target = get_closest_node(G, x_target, y_target)
print("source:", source)
print("target:", target)

# add weights into graph
for i in range(len(pois_df)):
    for j in range(len(pois_df)):
        weight = weights_matrix[i, j]
        if weight != np.inf:
            G.add_edge(i, j, weight=weight)

# find the shortest_path and check it
k = 100
for i in range(k):
    shortest_path = nx.shortest_path(G, source=source, target=target, weight='weight', method="dijkstra")
    unfeasible = check_path(shortest_path)
    print(f"shortest_path {shortest_path}, unfeasible is {unfeasible}")
    if unfeasible:
        print(f"shortest_path {shortest_path}, unfeasible is {unfeasible}")
        # randomly delete a vertex in the shortest path
        if len(shortest_path) > 2:
            remove_node = random.choice(shortest_path[1:-1])
            print(remove_node)
            G.remove_node(remove_node)
        if i == k - 1:
            print(f"can not find a feasible path in {k}-shortest paths")
        continue
    total_cost = sum(G[shortest_path[i]][shortest_path[i + 1]]['weight'] for i in range(len(shortest_path) - 1))
    total_time = total_cost / 3600
    print(f"travling time = {total_time}h")
    break

visu(shortest_path)







