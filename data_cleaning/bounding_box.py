# Date: 2023-07-18
# Author: Xubin Zhang
# Description:Calculate the bounding box formed by source and target coordinates.

def bbox(source_lat, source_lon, target_lat, target_lon):
    # Calculate the North latitude, West longitude, South latitude, and East longitude
    south_lat = min(source_lat, target_lat)
    west_lon = min(source_lon, target_lon)
    north_lat = max(source_lat, target_lat)
    east_lon = max(source_lon, target_lon)

    return south_lat, west_lon, north_lat, east_lon


# # # Example: latitude and longitude of source and target points
# source_lat, source_lon, target_lat, target_lon = 49.0130, 8.4093, 52.5253, 13.3694
#
# # create bounding box
# bbox = bbox(source_lat, source_lon, target_lat, target_lon)
# #bbox = f"({south_lat},{west_lon},{north_lat},{east_lon})"
#
# # test
# print(bbox)