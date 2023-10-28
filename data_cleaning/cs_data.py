# Date: 2023-07-18
# Author: Xubin Zhang
# Description: Add altitude in Ladesaeulenregister-processed.xlsx, fix wrong format of latitude and longitude, fix wrong latitude and longitude of locations 
# The accuracy of some coordinates is low, and the altitude cannot be obtained, e.g. lat=48,lng=10 is a location with low accuracy
# 1.Read the first rows_num rows of the file,
# 2.obtain the altitude through the openrouteservice api for each location,
# 3.store the location with altitude in cs_data.csv
# 4.delete the first rows_num rows of the original file
# input : Ladesaeulenregister-processed.xlsx
# output : cs_data.csv

import pandas as pd
import requests
import time
import os
import sys


# file_path = 'Ladesaeulenregister-processed.xlsx'
# rows_num = 1000
# # Maximum number of retries for API
# max_retries = 2

api_key = sys.argv[1]
file_path = sys.argv[2]
rows_num = int(sys.argv[3])
# Maximum number of retries for API
max_retries = int(sys.argv[4])

# print("API Key:", api_key)
print("File Path:", file_path)
print("Rows Num:", rows_num)
print("Max Retries:", max_retries)

# Get altitude for each location
def get_elevation(latitude, longitude):
    base_url = "https://api.openrouteservice.org/elevation/point"
    params = {
        "api_key": api_key,
        "geometry": f"{longitude},{latitude}",
    }

    wait = 120
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

    return None

def cs_data(file_path):

    # read file
    df = pd.read_excel(file_path)
    processed_rows = []  # List to store successfully processed row indices

    #Get the latitude and longitude of the first rows_num rows,
    #get the altitude information and save it in new column
    for i, row in df.iloc[:rows_num].iterrows():

        elevation = get_elevation(row["Latitude"], row["Longitude"])

        if elevation is not None:
            df.at[i, "Elevation"] = elevation
            processed_rows.append(i)
        else:
            # failed
            print("failed")

            if os.path.exists('../cs_data.csv'):
                # Append the updated rows_num data from df_tem to parking_bbox.csv
                df.iloc[processed_rows].to_csv('cs_data.csv', mode='a', index=False, header=False)
            else:
                # If cs_data.csv does not exist, create a new file with header
                df.iloc[processed_rows].to_csv('cs_data.csv', mode='a', index=False)

            # Drop the processed rows from df
            df = df.drop(processed_rows, axis=0)

            if df.empty:
                print("All altitudes have been obtained")
                os.remove(file_path)
                print("Deleted the file.")
            else:
                df.to_excel(file_path, index=False)
                print("Done, partial altitudes have been obtained")
            return None

        time.sleep(1)  # Wait 1 second after each request

    # Check the existing 'parking_data.csv' file
    if os.path.exists('../cs_data.csv'):
        # Append the updated rows_num data from df_tem to parking_bbox.csv
        df.iloc[:rows_num].to_csv('cs_data.csv', mode='a', index=False, header=False)
    else:
        # If cs_data.csv does not exist, create a new file with header
        df.iloc[:rows_num].to_csv('cs_data.csv', mode='a', index=False)

    #Delete the first rows_num rows of the original table
    df = df.iloc[rows_num:]

    if df.empty:
        print("All altitudes have been obtained")
        os.remove(file_path)
        print("Deleted the file.")
    else:
        df.to_excel(file_path, index=False)
        print("Done, partial altitudes have been obtained")

    return None


cs_data(file_path)
