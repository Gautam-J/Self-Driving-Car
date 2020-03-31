import os
import numpy as np

'''
Used to separate the road images alone from final_data.npy
'''

final_data_dir = "C:\\Users\\gauta\\python projects\\self-driving-car\\data\\final_data.npy"
org_data = np.load(final_data_dir, allow_pickle=True)

road_data = np.array([i for i, _, _ in org_data])
print(road_data.shape)

if not os.path.exists('data'):
    os.makedirs('data')

np.save('data\\road_data.npy', road_data)
print('Road data saved!')
