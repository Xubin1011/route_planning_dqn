# Date: 2023-07-24
# Author: Xubin Zhang
# Description: Check duplicate rows in csv file



# Description: Check duplicate rows of parking

# import pandas as pd
#
# # Read the CSV file
# df = pd.read_csv('parking_location.csv')
#
# # Find duplicate rows based on 'Latitude' and 'Longitude'
# duplicate_coords = df[df.duplicated(['Latitude', 'Longitude'], keep=False)]
#
# # Output the duplicate rows
# if duplicate_coords.shape[0] == 0:
#     print("No duplicate rows found.")
# else:
#     print("Duplicate rows found:")
#     print(duplicate_coords)
#
# # Remove duplicate rows
# df.drop_duplicates(subset=['Latitude', 'Longitude'], inplace=True)
#
# # Save the DataFrame without duplicate rows to a new CSV file
# df.to_csv('parking_location_noduplicate.csv', index=False)
# print("done.")




# Description: Check duplicate rows of charging stations

# import pandas as pd
#
# # Read CSV file
# df = pd.read_csv('cs_filtered_02.csv')
#
# # Find duplicate rows
# duplicate_rows = df[df.duplicated()]
#
# if duplicate_rows.shape[0] == 0:
#     print("No duplicate rows found.")
# else:
#     print("Duplicate rows found:")
#     print(duplicate_rows)
#
# # Remove duplicate rows
# df.drop_duplicates(inplace=True)
#
# # Save the DataFrame back to the CSV file
# df.to_csv('cs_filtered_02_noduplicate.csv', index=False)
# print("done")




# Description: Extract needed info of CS

# import pandas as pd
#
# # Read the CSV file
# df = pd.read_csv('cs_filtered_02_noduplicate.csv')
#
# # Extract 'Latitude', 'Longitude', and 'Max_power' columns
# selected_columns = df[['Latitude', 'Longitude', 'Max_power']]
#
# # Save the selected columns to a new CSV file
# selected_columns.to_csv('cs_filtered_02_noduplicate_01.csv', index=False)





# Description: Check duplicate rows after extract

# import pandas as pd
#
# df = pd.read_csv('cs_filtered_02_noduplicate_01.csv')
# duplicate_rows = df[df.duplicated()]
# if duplicate_rows.shape[0] == 0:
#     print("No duplicate rows found.")
# else:
#     print("Duplicate rows found:")
#     print(duplicate_rows)
#
# # Remove duplicate rows
# df.drop_duplicates(inplace=True)
#
# df.to_csv('cs_filtered_02_noduplicate_02.csv', index=False)
# print("done")



# Description: Check duplicate based on 'Latitude' and 'Longitude'

# import pandas as pd
#
# # Read the CSV file
# df = pd.read_csv('cs_filtered_02_noduplicate_02.csv')
#
# # Find duplicate rows based on 'Latitude' and 'Longitude'
# duplicate_coords = df[df.duplicated(['Latitude', 'Longitude'], keep=False)]
#
# # Output the duplicate rows
# if duplicate_coords.shape[0] == 0:
#     print("No duplicate rows found.")
# else:
#     print("Duplicate rows found:")
#     print(duplicate_coords)





# Find same 'Latitude' and 'Longitude',
# save the row with the maximum 'Max_power' value

# import pandas as pd
#
# # Read the CSV file
# df = pd.read_csv('cs_filtered_02_noduplicate_02.csv')
#
# # Find duplicate rows based on 'Latitude' and 'Longitude'
# duplicate_coords = df[df.duplicated(['Latitude', 'Longitude'], keep=False)]
#
# # Find the row with the maximum 'Max_power' value among duplicates
# max_power_rows = duplicate_coords.loc[duplicate_coords.groupby(['Latitude', 'Longitude'])['Max_power'].idxmax()]
#
# # Drop duplicate rows and keep the row with the maximum 'Max_power' value
# df.drop_duplicates(subset=['Latitude', 'Longitude'], keep=False, inplace=True)
#
# # Concatenate the rows with the maximum 'Max_power' value and rows with unique coordinates
# result_df = pd.concat([df, max_power_rows])
#
# # Save the final result to a new CSV file
# result_df.to_csv('cs_filtered_02_noduplicate_final.csv', index=False)
# print("done")


#check again

import pandas as pd
#
# # Read the CSV file
# df = pd.read_csv('parking_bbox.csv')
#
# # Find duplicate rows based on 'Latitude' and 'Longitude'
# duplicate_coords = df[df.duplicated(['Latitude', 'Longitude', 'Altitude'], keep=False)]
#
# # Output the duplicate rows
# if duplicate_coords.shape[0] == 0:
#     print("No duplicate rows found.")
# else:
#     print("Duplicate rows found:")
#     print(duplicate_coords)
#     # Delete the duplicate rows
#     df.drop_duplicates(subset=['Latitude', 'Longitude', 'Altitude'], keep='first', inplace=True)
#     # Save the DataFrame back to the CSV file, overwriting the original file
#     df.to_csv('parking_bbox.csv', index=False)
#     print("Duplicate rows have been deleted")


# comper two files
def check_locations_exist(file1, file2):
    # Read the CSV files into DataFrames
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Create a set of tuples containing (Latitude, Longitude) from df2
    locations_set = set(tuple(x) for x in df2[['Latitude', 'Longitude']].values)

    # Initialize a counter for non-existing locations
    non_existing_count = 0

    # Check if each location in df1 exists in df2
    for index, row in df1.iterrows():
        latitude = row['Latitude']
        longitude = row['Longitude']

        if (latitude, longitude) not in locations_set:
            non_existing_count += 1

    print(f"The number of rows with locations that do not exist in table 2 is: {non_existing_count}")

    # if (latitude, longitude) in locations_set:
    #         print(f"Location ({latitude}, {longitude}) exists in both tables.")
    #     else:
    #         print(f"Location ({latitude}, {longitude}) does not exist in table 2.")

    return None


file1 = 'parking_bbox.csv'
file2 = 'F:\OneDrive\Thesis\Code\parking_bbox.csv'
check_locations_exist(file1, file2)







