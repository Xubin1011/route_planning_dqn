# Date: 7/27/2023
# Author: Xubin Zhang
# Description: Clean data, remove duplicate locations, extract the needed data
# (latitude, longitude, max_charging_power) from cs_data.csv
# (Die Liste beinhaltet die Ladeeinrichtungen aller Betreiberinnen und Betreiber,
# die das Anzeigeverfahren der Bundesnetzagentur vollständig abgeschlossen und einer Veröffentlichung im Internet zugestimmt haben.)
#input: cs_data.csv: latitude, longitude, power, altitude and unneeded info in Germany.
#output: cs_combo.csv: charging stations with combo type in Germany. latitude, longitude, altitude, power.
# cs_type2_combo.csv:charging stations with combo and type 2 in Germany. latitude, longitude, altitude, power.


import pandas as pd
import os

types_replaced_0 = "cs_data.csv"   # Unneeded types of sockets will be replaced to 0
average_power = 150  # The maximum charging power of an ebus (in kW)
combo = 1 # Only consider charging stations with combo type
type2_combo = 1 # Consider charging stations with combo type and type2 type


# Calculate unique types of sockets, output types.txt
# file_path_check_types = "Ladesaeulenregister-processed.xlsx"  # Calculate unique types of sockets
# file_path_check_types = "cs_filtered_03.csv"  # Calculate unique types of sockets
def check_types(file_path_check_types):

    # Read the data from the Excel/csv file
    data = pd.read_csv(file_path_check_types)
    #data = pd.read_excel(file_path_check_types)

    # Calculate unique values and their occurrences in 'Socket1', 'Socket2', 'Socket3', 'Socket4'
    #socket_columns = ['Socket_1', 'Socket_2', 'Socket_3', 'Socket_4']
    socket_columns = ['Rated_output', 'Max_socket_power', 'Max_power']


    unique_values = {}
    for column in socket_columns:
        unique_values[column] = data[column].value_counts()

    # Print and save the results to 'types.txt'
    with open('../power.txt', 'w', encoding='utf-8') as f:
        for column, values_counts in unique_values.items():
            f.write(f"Unique values and their occurrences in {column} column:\n")
            f.write(f"{values_counts}\n\n")

    print("types checked")
    return None

# check_types(file_path_check_types)



# Unneeded types of sockets will be replaced by 0
# output type2_combo_kept.csv

def keep_type2_combo(types_replaced_0):
    # Read the data from the Excel file
    data = pd.read_csv(types_replaced_0)

    # keep Type 2 and combo in a table
    # Replace values in 'Socket_1' column
    replace_values_socket1 = ['AC Schuko','AC CEE 5 polig', 'AC Schuko, AC CEE 5 polig']
    data['Socket_1'] = data['Socket_1'].replace(replace_values_socket1, 0)
    # Replace values in 'Socket_2' column
    replace_values_socket2 = ['AC Schuko', 'DC CHAdeMO', 'AC CEE 5 polig','AC Schuko, AC CEE 5 polig']
    data['Socket_2'] = data['Socket_2'].replace(replace_values_socket2, 0)
    # Replace values in 'Socket_3' column
    replace_values_socket3 = ['AC Schuko', 'DC CHAdeMO']
    data['Socket_3'] = data['Socket_3'].replace(replace_values_socket3, 0)
    # Replace values in 'Socket_4' column
    replace_values_socket4 = ['AC Schuko']
    data['Socket_4'] = data['Socket_4'].replace(replace_values_socket4, 0)
    # Save the modified data to a new CSV file
    data.to_csv('type2_combo_kept.csv', index=False)

    # Save the modified data to a new CSV file
    data.to_csv('type2_combo_kept.csv', index=False)

    print("replaced by 0, done")

    return None



# Unneeded types of sockets will be replaced by 0
# output combo_kept.csv

def keep_combo(types_replaced_0):
    # Read the data from the Excel file
    data = pd.read_csv(types_replaced_0)

    # keep only combo in a table
    # Replace values in 'Socket_1' column
    replace_values_socket1 = ['AC Steckdose Typ 2', 'AC Kupplung Typ 2', 'AC Steckdose Typ 2, AC Schuko', 'AC Steckdose Typ 2, AC Kupplung Typ 2',
                              'AC Kupplung Typ 2, AC Schuko', 'AC Steckdose Typ 2, AC Kupplung Typ 2, AC Schuko', 'AC Steckdose Typ 2, AC CEE 5 polig',
                              'AC Steckdose Typ 2, AC CEE 3 polig', 'AC Kupplung Typ 2, DC CHAdeMO', 'AC Steckdose Typ 2, AC Kupplung Typ 2, AC CEE 5 polig',
                              'AC Kupplung Typ 2, AC CEE 5 polig','AC Steckdose Typ 2, DC CHAdeMO', 'AC Schuko', 'AC Steckdose Typ 2, AC Schuko, AC CEE 5 polig',
                              'AC Kupplung Typ 2, AC CEE 3 polig', 'AC Steckdose Typ 2, AC Kupplung Typ 2, DC CHAdeMO', 'AC Steckdose Typ 2, CEE-Stecker',
                              'AC CEE 5 polig','AC Steckdose Typ 2, AC Kupplung Typ 2, AC Schuko, AC CEE 3 polig; AC CEE 5 polig', 'AC Kupplung Typ 2, Adapter Typ1  Auto auf Typ2 Fahrzeugkupplung',
                              'AC Steckdose Typ 2, AC Kupplung Typ 2, AC Schuko, DC CHAdeMO', 'AC Schuko, AC CEE 5 polig', 'AC Steckdose Typ 2, AC Kupplung Typ 2, AC Schuko, AC CEE 5 polig',
                              'AC Steckdose Typ 2, AC Schuko, DC CHAdeMO']
    data['Socket_1'] = data['Socket_1'].replace(replace_values_socket1, 0)
    # Replace values in 'Socket_2' column
    replace_values_socket2 = ['AC Steckdose Typ 2', 'AC Steckdose Typ 2, AC Schuko', 'AC Kupplung Typ 2', 'AC Steckdose Typ 2, AC Kupplung Typ 2', 'AC Kupplung Typ 2, DC CHAdeMO',
                              'AC Schuko', 'AC Steckdose Typ 2, AC CEE 5 polig', 'AC Steckdose Typ 2, AC Kupplung Typ 2, DC CHAdeMO',
                              'AC Steckdose Typ 2, AC CEE 3 polig', 'AC Kupplung Typ 2, AC Schuko', 'DC CHAdeMO', 'AC Steckdose Typ 2, AC Kupplung Typ 2, AC Schuko',
                              'AC Steckdose Typ 2, AC Kupplung Typ 2, AC CEE 5 polig', 'AC Kupplung Typ 2, AC CEE 3 polig', 'AC Steckdose Typ 2, DC CHAdeMO',
                              'AC Kupplung Typ 2, AC CEE 5 polig', 'AC CEE 5 polig', 'AC Steckdose Typ 2, AC Kupplung Typ 2, AC Schuko, AC CEE 3 polig; AC CEE 5 polig',
                              'AC Schuko, AC CEE 5 polig', 'AC Steckdose Typ 2, AC Kupplung Typ 2, AC / CEE']
    data['Socket_2'] = data['Socket_2'].replace(replace_values_socket2, 0)
    # Replace values in 'Socket_3' column
    replace_values_socket3 = ['AC Steckdose Typ 2', 'AC Kupplung Typ 2', 'AC Kupplung Typ 2, DC CHAdeMO', 'AC Steckdose Typ 2, AC Kupplung Typ 2', 'AC Steckdose Typ 2, AC Schuko',
                              'AC Steckdose Typ 2, AC Kupplung Typ 2, DC CHAdeMO', 'AC Schuko', 'DC CHAdeMO', 'AC Steckdose Typ 2, AC CEE 3 polig', 'AC Steckdose Typ 2, AC CEE 5 polig',
                              'AC Kupplung Typ 2, AC CEE 5 polig', 'AC Steckdose Typ 2, DC CHAdeMO', 'AC Kupplung Typ 2, AC Schuko', 'AC Steckdose Typ 2, AC Kupplung Typ 2, AC Schuko']
    data['Socket_3'] = data['Socket_3'].replace(replace_values_socket3, 0)
    # Replace values in 'Socket_4' column
    replace_values_socket4 = ['AC Steckdose Typ 2', 'AC Kupplung Typ 2', 'AC Schuko', 'AC Steckdose Typ 2, AC Kupplung Typ 2',
                              'AC Steckdose Typ 2, AC Schuko', 'AC Kupplung Typ 2, AC CEE 5 polig', 'AC Steckdose Typ 2, AC Kupplung Typ 2, AC Schuko, AC CEE 3 polig', 'AC Steckdose Typ 2, AC CEE 5 polig']
    data['Socket_4'] = data['Socket_4'].replace(replace_values_socket4, 0)
    # Save the modified data to a new CSV file
    data.to_csv('combo_kept.csv', index=False)

    print("replaced by 0, done")

    return None


# Only keep needed info in a table (all charging station in DE) (Latitude, Longitude, Altitude,Average charging power)
# input: type2_combo_kept.csv or combo_kept.csv
# output: cs_combo.csv or cs_type2_combo.csv
# 1. Replace None by 0
# 2. Replace 'P1', 'P2', 'P3', 'P4' values with 0 when Socket values are 0
# 3. If all values in Socket columns are 0 or '0', drop those rows. That means delete rows that needed types does not exist
# 4. Compare the max. output power of sockets with rated power of charging station, keep the min. value of them as max.charging power
# 5. Compare max.charging power with average charging power, keep the min. value of them as charging power
# that means, if max.charging power is bigger than average charging power, use the average charging power(100 KW),
# otherwise, use the max.charging power, there are


def combo_data_clean(file_path_clean):
    # Read the data from the CSV file
    data = pd.read_csv(file_path_clean)

    # Fill empty values with 0
    data.fillna(0, inplace=True)

    # Replace 'P1', 'P2', 'P3', 'P4' values with 0 when Socket values are 0
    data['P1'] = data['P1'].mask(data['Socket_1'].isin([0, '0']), 0).astype(int)
    data['P2'] = data['P2'].mask(data['Socket_2'].isin([0, '0']), 0).astype(int)
    data['P3'] = data['P3'].mask(data['Socket_3'].isin([0, '0']), 0).astype(int)
    data['P4'] = data['P4'].mask(data['Socket_4'].isin([0, '0']), 0).astype(int)

    #if all values in Socket columns are 0 or '0', drop those rows
    socket_columns = ['Socket_1', 'Socket_2', 'Socket_3', 'Socket_4']
    #data = data[(data['Socket_1'] != 0) | (data['Socket_2'] != 0) | (data['Socket_3'] != 0) | (data['Socket_4'] != 0)]
    data = data[~((data[socket_columns] == '0') | (data[socket_columns] == 0)).all(axis=1)]

    # Compare power column with Rated_output column and select charging power
    data['Max_socket_power'] = data[['P1', 'P2', 'P3', 'P4']].max(axis=1)
    data['Max_power'] = data[['Max_socket_power', 'Rated_output']].min(axis=1)
    data['Power'] = data['Max_power'].clip(upper=average_power)

    # Select needed ifo and save to cs_combo.csv file
    data[['Latitude', 'Longitude', 'Elevation', 'Power']].to_csv("cs_combo.csv", index=False)

    # Output the number of rows in the table
    num_rows = data.shape[0]
    print("Number of cs with combo:", num_rows)

    return None


def type2_combo_data_clean(file_path_clean):
    # Read the data from the CSV file
    data = pd.read_csv(file_path_clean)

    # Fill empty values with 0
    data.fillna(0, inplace=True)

    # Replace 'P1', 'P2', 'P3', 'P4' values with 0 when Socket values are 0
    data['P1'] = data['P1'].mask(data['Socket_1'].isin([0, '0']), 0).astype(int)
    data['P2'] = data['P2'].mask(data['Socket_2'].isin([0, '0']), 0).astype(int)
    data['P3'] = data['P3'].mask(data['Socket_3'].isin([0, '0']), 0).astype(int)
    data['P4'] = data['P4'].mask(data['Socket_4'].isin([0, '0']), 0).astype(int)

    #if all values in Socket columns are 0 or '0', drop those rows
    socket_columns = ['Socket_1', 'Socket_2', 'Socket_3', 'Socket_4']
    #data = data[(data['Socket_1'] != 0) | (data['Socket_2'] != 0) | (data['Socket_3'] != 0) | (data['Socket_4'] != 0)]
    data = data[~((data[socket_columns] == '0') | (data[socket_columns] == 0)).all(axis=1)]

    # Compare power column with Rated_output column and select charging power
    data['Max_socket_power'] = data[['P1', 'P2', 'P3', 'P4']].max(axis=1)
    data['Max_power'] = data[['Max_socket_power', 'Rated_output']].min(axis=1)
    data['Power'] = data['Max_power'].clip(upper=average_power)

    # Select needed ifo and save to cs_combo.csv file
    data[['Latitude', 'Longitude', 'Elevation', 'Power']].to_csv("cs_type2_combo.csv", index=False)

    # Output the number of rows in the table
    num_rows = data.shape[0]
    print("Number of cs with type2_combo:", num_rows)

    return None

def check_duplicates(file_path_duplicate):

    df = pd.read_csv(file_path_duplicate)

    # Find duplicate rows
    duplicate_coords = df[df.duplicated(['Latitude', 'Longitude', 'Elevation', 'Power'], keep=False)]

    # Output the duplicate rows
    if duplicate_coords.shape[0] == 0:
        print("No duplicate rows found.")
    else:
        print("Duplicate rows found in", file_path_duplicate, ":")
        print(duplicate_coords)
        # Delete the duplicate rows
        df.drop_duplicates(subset=['Latitude', 'Longitude', 'Elevation', 'Power'], keep='first', inplace=True)
        # Save back to the CSV file, overwriting the original file
        df.to_csv(file_path_duplicate, index=False)
        print("Duplicate rows have been deleted")
    return None



if combo == 1:

    keep_combo(types_replaced_0)
    file_path_clean = "combo_kept.csv"
    combo_data_clean(file_path_clean)
    os.remove(file_path_clean)
    file_path_duplicate = "../cs_combo.csv"
    check_duplicates(file_path_duplicate)

if type2_combo == 1:

    keep_type2_combo(types_replaced_0)
    file_path_clean = "type2_combo_kept.csv"
    type2_combo_data_clean(file_path_clean)
    os.remove(file_path_clean)
    file_path_duplicate = "../cs_type2_combo.csv"
    check_duplicates(file_path_duplicate)

print("Data has been cleaned")






