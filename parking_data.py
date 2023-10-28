# Date: 7/27/2023
# Author: Xubin Zhang
# Description: Get the latitude, longitude, altitude of parking lots within the bbox range, and output the csv table
#input： latitude and longitude of two pois
# parking_data.csv: include the latitude, longitude and elevation of all locations that has benn used.
#output:
# parking_bbox.csv: include the latitude, longitude and elevation of locations within bbox
# parking_bbox_tem.csv:Do not modify, include the latitude and longitude of the new location.
# It will be automatically deleted after all altitudes are obtained.


import requests
import pandas as pd
import time
import os


max_retries = 2 # Maximum number of retries for elevation API
api_key = "5b3ce3597851110001cf624880a184fac65b416298dee8f52e43a0fe"
rows_num = 5
source_lat, source_lon, target_lat, target_lon = 49.0130, 8.4093, 52.5253, 13.3694 #kit to berlin
# source_lat, source_lon, target_lat, target_lon = 48.0130, 7.4093, 51.458, 12.3694

def bounding_box(source_lat, source_lon, target_lat, target_lon):
    # Calculate the North latitude, West longitude, South latitude, and East longitude
    south_lat = min(source_lat, target_lat)
    west_lon = min(source_lon, target_lon)
    north_lat = max(source_lat, target_lat)
    east_lon = max(source_lon, target_lon)

    return south_lat, west_lon, north_lat, east_lon


#get locations within bbox and store in parking_bbox_tem.csv
def overpass_query(query):
    overpass_url = "https://overpass-api.de/api/interpreter"
    response = requests.get(overpass_url, params={"data": query})
    return response.json()

def get_parking_rest_area_services_data(bbox):
    query = f"""
    [out:json];
    (
        node["amenity"="parking"]["access"="yes"]{bbox};
        node["highway"="rest_area"]{bbox};
        node["highway"="services"]{bbox};
    );
    out;
    """
    response_json = overpass_query(query)
    data = []
    for element in response_json["elements"]:
        if element["type"] == "node":
            lat = element["lat"]
            lon = element["lon"]
            data.append((lat, lon))

    # Create DataFrame to store the locations
    parking_bbox_tem = pd.DataFrame(data, columns=["Latitude", "Longitude"])

    # Save to CSV
    parking_bbox_tem.to_csv("parking_bbox_tem.csv", index=False)

    return None


def check_duplicates(file_path_duplicate):

    df = pd.read_csv(file_path_duplicate)

    # Find duplicate rows based on 'Latitude''Longitude''Altitude'
    duplicate_coords = df[df.duplicated(['Latitude', 'Longitude', 'Altitude'], keep=False)]

    # Output the duplicate rows
    if duplicate_coords.shape[0] == 0:
        print("No duplicate rows found.")
    else:
        print("Duplicate rows found in", file_path_duplicate, ":")
        print(duplicate_coords)
        # Delete the duplicate rows
        df.drop_duplicates(subset=['Latitude', 'Longitude', 'Altitude'], keep='first', inplace=True)
        # Save back to the CSV file, overwriting the original file
        df.to_csv(file_path_duplicate, index=False)
        print("Duplicate rows have been deleted")
    return None


def get_elevation(latitude, longitude):
    base_url = "https://api.openrouteservice.org/elevation/point"
    params = {
        "api_key": api_key,
        "geometry": f"{longitude},{latitude}",
    }

    wait =300

    # Retry loop
    for retry in range(max_retries):

        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            try:
                data = response.json()
                elevation = data["geometry"]["coordinates"][2]
                return elevation

            except KeyError:
                print("Unexpected response format. Could not find 'geometry' key in the response.")
                print(f"Response content: {response.text}")
                print("wait", wait, "s")
                time.sleep(wait)  # Wait 300 second before retrying
                wait += 60

            except Exception as e:
                print(f"Error occurred while processing the API response: {e}")
                print(f"Response content: {response.text}")
                print("wait", wait, "s")
                time.sleep(wait)  # Wait 60 second before retrying
                wait += 60

        else:
            print(f"Failed to fetch elevation data. Status code: {response.status_code}")
            print(f"Retry attempt: {retry + 1}")
            time.sleep(60)  # Wait 60 second before retrying

    return None


#Get elevation for new locations, and add to parking_bbox.csv, parking_data.csv,
#delete the first rows_num rows of parking_bbox_tem.csv
def parking_bbox_tem_altitude():
    # read file
    df_tem = pd.read_csv("parking_bbox_tem.csv")

    if df_tem.empty:
        print("All altitudes have been obtained")
        os.remove('parking_bbox_tem.csv')
        print("The parking_bbox_tem.csv file is empty. Deleted the file.")

    else:
        # Get the latitude and longitude of the first rows_num rows,
        # get the altitude information and save it in the third column
        for i, row in df_tem.iloc[:rows_num].iterrows():
            df_tem.at[i, "Altitude"] = get_elevation(row["Latitude"], row["Longitude"])
            time.sleep(1)  # Wait 1 second after each request

        # Read the existing 'parking_bbox.csv' file
        if os.path.exists('parking_bbox.csv'):
            df_bbox = pd.read_csv("parking_bbox.csv")
        else:
            df_bbox = pd.DataFrame(columns=['Latitude', 'Longitude', 'Altitude'])

        # Read the existing 'parking_data.csv' file
        if os.path.exists('parking_data.csv'):
            df_data = pd.read_csv("parking_data.csv")
        else:
            df_data = pd.DataFrame(columns=['Latitude', 'Longitude', 'Altitude'])

        # Append the updated rows_num data from df_tem to df_bbox
        # df_bbox = df_bbox.append(df_tem.iloc[:rows_num], ignore_index=True)
        # df_data = df_data.append(df_tem.iloc[:rows_num], ignore_index=True)
        df_bbox = pd.concat([df_bbox, df_tem.iloc[:rows_num]], ignore_index=True)
        df_data = pd.concat([df_data, df_tem.iloc[:rows_num]], ignore_index=True)

        # Save to 'parking_data.csv' file
        df_bbox.to_csv("parking_bbox.csv", index=False)
        df_data.to_csv("parking_data.csv", index=False)

        # Delete the first rows_num rows of the original table
        df_tem = df_tem .iloc[rows_num:]
        df_tem.to_csv("parking_bbox_tem.csv", index=False)

        # Check if df_tem is empty
        if df_tem.empty:
            print("All altitudes have been obtained")
            os.remove('parking_bbox_tem.csv')
            print("The parking_bbox_tem.csv file is empty. Deleted the file.")
            check_duplicates('parking_bbox.csv')
            check_duplicates('parking_data.csv')
        else:
            # Get the number of rows after deletion
            num_rows_remaining = df_tem.shape[0]
            print("remaining", num_rows_remaining, "rows in parking_bbox_tem.csv")

    return None


#Get altitude from parking_data.csv
#Delete matched location in parking_bbox_tem.csv
#Store the matched locations in parking_bbox.csv with altitude
def compare_and_update_parking_data():
    # Check if parking_data.csv file exists
    if os.path.exists('parking_data.csv'):
        # Read the parking_data.csv file
        parking_data = pd.read_csv('parking_data.csv')
    else:
        # If the file does not exist, create a new DataFrame and add the header
        parking_data = pd.DataFrame(columns=['Latitude', 'Longitude', 'Altitude'])
        parking_data.to_csv('parking_data.csv', index=False)

    # read parking_bbox_tem.csv
    parking_bbox_tem = pd.read_csv('parking_bbox_tem.csv')

    # store the matched locations
    matching_rows_list = []

    # search the matched locations between parking_bbox_tem.csv and parking_data.csv
    for index, row in parking_bbox_tem.iterrows():
        latitude = row['Latitude']
        longitude = row['Longitude']

        # store the matched locations from parking_data.csv into matching_rows
        matching_rows = parking_data.loc[(parking_data['Latitude'] == latitude) & (parking_data['Longitude'] == longitude)]

        if not matching_rows.empty:
            # store in list
            matching_rows_list.append(matching_rows)

    if matching_rows_list:
        all_matching_rows = pd.concat(matching_rows_list)
        # store the matched locations in parking_bbox.csv
        #all_matching_rows.to_csv('parking_bbox.csv', mode='a', index=False, header=False)
        if os.path.exists('parking_bbox.csv'):
            # Append the updated rows_num data from df_tem to parking_bbox.csv
            all_matching_rows.to_csv('parking_bbox.csv', mode='a', index=False, header=False)
        else:
            # If parking_bbox.csv does not exist, create a new file with header
            all_matching_rows.to_csv('parking_bbox.csv', mode='a', index=False)

        # delete the matched locations in parking_bbox_tem.csv
        merged_df = parking_bbox_tem.merge(all_matching_rows, on=['Latitude', 'Longitude'], how='left', indicator=True)
        tem_filtered = merged_df[merged_df['_merge'] == 'left_only'].drop(columns='_merge')
        tem_filtered.to_csv('parking_bbox_tem.csv', index=False)

    return None



# Check if parking_bbox_tem.csv file exists
if os.path.exists('parking_bbox_tem.csv'):

    # Get altitude
    parking_bbox_tem_altitude()

else:

    # If the file does not exist,get new location within bbox
    bbox = bounding_box(source_lat, source_lon, target_lat, target_lon)

    #get locations within bbox and store in parking_bbox_tem.csv
    get_parking_rest_area_services_data(bbox)

    #get altitude from parking_data.csv
    #and delete matched location in parking_bbox_tem.csv
    #store the matched locations in parking_bbox.csv with altitude
    compare_and_update_parking_data()

    # Get altitude for new locations
    parking_bbox_tem_altitude()

print("done")









