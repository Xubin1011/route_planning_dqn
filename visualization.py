# Date: 7/25/2023
# Author: Xubin Zhang
# Description: This file contains the implementation of...

import pandas as pd
import folium
from data_cleaning.bounding_box import bbox

def visualization(file1, file2, file3, source_lat, source_lon, target_lat, target_lon, map_name):
    # calculate the bounding box
    south_lat, west_lon, north_lat, east_lon = bbox(source_lat, source_lon, target_lat, target_lon)

    # Read data from file1
    data1 = pd.read_csv(file1)

    # Read data from file2
    data2 = pd.read_csv(file2)

    # Calculate the center of the bounding box
    center_lat = (south_lat + north_lat) / 2
    center_lon = (west_lon + east_lon) / 2

    # Create a map object
    map_object = folium.Map(location=[center_lat, center_lon], zoom_start=10)

    # Add the bounding box area to the map
    bbox_coords = [[south_lat, west_lon], [north_lat, west_lon], [north_lat, east_lon], [south_lat, east_lon], [south_lat, west_lon]]
    folium.Polygon(locations=bbox_coords, color='blue', fill=True, fill_color='blue', fill_opacity=0.2).add_to(map_object)

    # Read data from file1 and add yellow markers for data points
    for _, row in data1.iterrows():
        # latitude, longitude = row['Latitude'], row['Longitude']
        # popup_content = 'Charging Station'
        # tooltip_content = f'Latitude: {latitude}, Longitude: {longitude}'
        # folium.Marker(location=[latitude, longitude], popup=popup_content, tooltip=tooltip_content, icon=folium.Icon(color='purple')).add_to(map_object)

        latitude, longitude = row['Latitude'], row['Longitude']
        folium.CircleMarker(location=[latitude, longitude], radius=1, color='yellow', fill=True, fill_color='yellow').add_to(map_object)

    # Read data from file2 and add blue markers for data points
    for _, row in data2.iterrows():
        latitude, longitude = row['Latitude'], row['Longitude']
        folium.CircleMarker(location=[latitude, longitude], radius=1, color='blue', fill=True, fill_color='blue').add_to(map_object)

    # Read data from file3 as path coordinates
    path_data = pd.read_csv(file3)
    path_coords = list(zip(path_data['Latitude'], path_data['Longitude']))

    for coord in path_coords:
        latitude, longitude = coord
        folium.Marker(location=[latitude, longitude],
                      popup=f'Latitude: {latitude}<br>Longitude: {longitude}',
                      icon=folium.Icon(color='green')).add_to(map_object)
        folium.CircleMarker(location=[latitude, longitude], radius=2, color='red', fill=True, fill_color='red').add_to(
            map_object)

    # Add a red line to represent the path
    folium.PolyLine(locations=path_coords, color='red').add_to(map_object)

    # Save the map as an HTML file and display it
    map_object.save(map_name)

# # test
# file1 = 'cs_combo_bbox.csv'
# file2 = 'parking_bbox.csv'
# file3 = 'path_coords.csv'
# map_name = "no_route.html"
# source_lat, source_lon, target_lat, target_lon = 49.01302968199333, 8.409265137665193, 52.52533075184041, 13.369384859383123
# visualization(file1, file2, file3, source_lat, source_lon, target_lat, target_lon, map_name)




