import pandas as pd

file_path_ch = 'cs_combo_bbox.csv'
file_path_p = 'parking_bbox.csv'
initial_data_ch = pd.read_csv("cs_combo_bbox.csv")
initial_data_p = pd.read_csv("parking_bbox.csv")
data_ch = initial_data_ch.copy()
data_p = initial_data_p.copy()

num_sucesse = 0
num_max_q = 1
total_num_select = 0