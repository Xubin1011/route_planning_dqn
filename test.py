from deploy_compare.way_deploy_cs import  way
from global_var_dij import file_path_ch

myway = way()
nearest_location = myway.nearest_location(file_path_ch, 51.2750583, 8.8710819, 6)
print(nearest_location)

# from consumption_duration import haversine
# dis = haversine()

