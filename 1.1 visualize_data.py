import numpy as np
import cv2

'''
Use this script to visualize the data captured along with the label.
'''

d = input('Enter the filename: ')
train_data = np.load(f'data\\{d}.npy', allow_pickle=True)

for data in train_data:
    screen = data[0]
    minimap = data[1]
    choice = data[2]

    cv2.imshow('screen', screen)
    cv2.imshow('minimap', minimap)
    print(choice)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
