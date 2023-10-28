import torch
from global_var_150 import initial_data_p, initial_data_ch
from consumption_duration import haversine

x_source = 49.0130  # source
y_source = 8.4093

# update coordinates of target, select the closest CH or P as target
min_dis = None
for index, row in initial_data_ch.iterrows():
    distance = haversine(row['Latitude'], row['Longitude'], x_source, y_source)
    if min_dis is None or distance < min_dis:
        min_dis = distance
        closest_index_ch = index
closest_point_ch = initial_data_ch.loc[closest_index_ch]
x_source_ch = closest_point_ch['Latitude']
y_source_ch = closest_point_ch['Longitude']
print("source_ch:", x_source_ch, y_source_ch, closest_index_ch)
min_dis = None
for index, row in initial_data_p.iterrows():
    distance = haversine(row['Latitude'], row['Longitude'], x_source, y_source)
    if min_dis is None or distance < min_dis:
        min_dis = distance
        closest_index_p = index
closest_point_p = initial_data_p.loc[closest_index_p]
x_source_p = closest_point_p['Latitude']
y_source_p = closest_point_p['Longitude']
print("source_p :", x_source_p, y_source_p, closest_index_p)
