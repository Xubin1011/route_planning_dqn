# Date: 2023-07-18
# Author: Xubin Zhang
# Description: Extract latitude and longitude within the specified bounding box range
# from cs_combo.csv or cs_type2_combo.csv
#input: cs_combo.csv or cs_type2_combo.csv
#output: cs_combo_bbox.csv or cs_type2_combo_bbox.csv. The charging stations within bbox.


import pandas as pd

combo = 1 # Only consider charging stations with combo type
type2_combo = 0 # Consider charging stations with combo type and type2 type
source_lat, source_lon, target_lat, target_lon = 49.0130, 8.4093, 52.5253, 13.3694 #kit to berlin

def bounding_box(source_lat, source_lon, target_lat, target_lon):
    # Calculate the North latitude, West longitude, South latitude, and East longitude
    south_lat = min(source_lat, target_lat)
    west_lon = min(source_lon, target_lon)
    north_lat = max(source_lat, target_lat)
    east_lon = max(source_lon, target_lon)

    return south_lat, west_lon, north_lat, east_lon

def cs_bbox(file_path, new_path):

    south_lat, west_lon, north_lat, east_lon = bounding_box(source_lat, source_lon, target_lat, target_lon)

    # Read data from excel file into a DataFrame
    df = pd.read_csv(file_path)

    # Filter the rows within the bounding box range
    filtered_df = df[
        (df['Latitude'] >= south_lat) & (df['Latitude'] <= north_lat) &
        (df['Longitude'] >= west_lon) & (df['Longitude'] <= east_lon)
    ]

    # Save the filtered result
    filtered_df.to_csv(new_path, index=False)

    return None

if combo == 1:

    file_path = "../cs_combo.csv"
    new_path = "../cs_combo_bbox.csv"
    cs_bbox(file_path, new_path)

if type2_combo == 1:

    file_path = "../cs_type2_combo.csv"
    new_path = "cs_type2_combo_bbox.csv"
    cs_bbox(file_path, new_path)

print("Extraction completed")

