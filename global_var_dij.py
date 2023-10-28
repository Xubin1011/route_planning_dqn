import pandas as pd

file_path_ch = '/home/utlck/PycharmProjects/Dij_results/dijkstra_pois.csv'
file_path_p = 'parking_bbox.csv'
initial_data_ch = pd.read_csv(file_path_ch)
initial_data_p = pd.read_csv("parking_bbox.csv")
data_ch = initial_data_ch.copy()
data_p = initial_data_p.copy()