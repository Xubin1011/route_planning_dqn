# Date: 2023-07-18
# Author: Xubin Zhang
# Description: Find the n nearest locations to the current position.
# Parameters:
    # file_path : Path of the CSV file, containing information of Latitude and Longitude
    # x1: Latitude of the current position
    # y1: Longitude of the current position
    # n: Number of nearest locations to find
#Returns: A list containing information of n nearest locations.


import pandas as pd
import heapq
# the haversine function from distance_haversine.py
from consumption_duration import haversine


def nearest_location(file_path, x1, y1, n):

    # Read the CSV file and extract latitude and longitude
    data = pd.read_csv(file_path)

    latitudes = data["Latitude"]
    longitudes = data["Longitude"]

    # Priority queue to store information of the n nearest locations
    closest_locations = []

    # Iterate through all the locations
    for lat, lon in zip(latitudes, longitudes):
        # Calculate the distance
        distance = haversine(x1, y1, lat, lon)

        if distance < 25000:
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
                heapq.heappop(closest_locations) #pop the farthest location
                heapq.heappush(closest_locations, (neg_distance, lat, lon)) #insert the closer location

    # convert the distance back to positive values
    closest_locations = pd.DataFrame(closest_locations, columns=["Neg_Distance", "Latitude", "Longitude"])
    closest_locations["Distance"] = -closest_locations["Neg_Distance"]
    closest_locations.drop(columns=["Neg_Distance"], inplace=True)

    # Sort by distance in ascending order
    closest_locations.sort_values(by="Distance", inplace=True)

    # Extract information of the n nearest locations
    nearest_locations = closest_locations.head(n).reset_index(drop=True)
    return nearest_locations

# # test
# # file_path = "cs_combo_bbox.csv"
# file_path = 'parking_bbox.csv'
# x1, y1 = 49.403861,9.390352
#
# n = 4
# nearest_locations = nearest_location(file_path, x1, y1, n)
# # nearest_locations.drop(0, inplace=True)
# print(nearest_locations)
# # nearest_locations.to_csv('nearest.csv', index=False, header=True)

