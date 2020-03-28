import numpy as np
import cv2
import time
from grabscreen import grab_screen
from getkeys import key_check
from countdown import CountDown
import os
import psutil

'''
Run this script to get training data. The script captures the frame image of
the main screen and the mini map, along with the button you press for that corresponding frame,
i.e., whether you drive straight, left, or right.

Since numpy arrays can become huge files, it's best to collect your data in
batches. If the length of data becomes greater than 100k (or until the script
takes a lot of time to save), start a new batch.
'''


def keys_to_output(keys):

    '''
    One hot encodes our label. I use 'wasd' to drive my car. Therefore, when I
    press 'a', it means I go left, and if I press 'd', I go right.

    One-Hot Encoding of labels:
    forward = [0, 1, 0]
    left = [1, 0, 0]
    right = [0, 0, 1]
    '''

    if 'A' in keys:
        output = [1, 0, 0]
    elif 'D' in keys:
        output = [0, 0, 1]
    else:
        output = [0, 1, 0]

    return output


def main():
    CountDown(5)

    if os.path.isfile(file_name):
        print('Existing Training Data:' + str(len(training_data)))
        print('Capturing Data!')
    else:
        print('Capturing Data Freshly!')

    paused = False
    while True:

        if not paused:
            screen = grab_screen(region=(270, 250, 650, 450))
            minimap = grab_screen(region=(100, 390, 230, 490))
            screen = cv2.resize(screen, (200, 80))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            minimap = cv2.resize(minimap, (50, 50))
            minimap = cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)

            keys = key_check()
            output = keys_to_output(keys)
            training_data.append([screen, minimap, output])

            # prints the size of the object every often so that RAM doesn't filled up
            if len(training_data) % 500 == 0:
                print('Saving data...')
                np.save(file_name, training_data)
                print(f'Saved {file_name} successfully!')
                print(f'System Memory Usage: {psutil.virtual_memory().percent} %')
                print(f'New Training Data: {len(training_data)} frames')
                print('-' * 80)

        keys = key_check()

        '''
        you can press 'e' while in game to pause/unpause the script from
        capturing data.
        '''
        if 'E' in keys:
            if paused:
                paused = False
                print('Unpaused!')
                time.sleep(1)
                print('Capturing Data!')
            else:
                paused = True
                print('Paused!')
                time.sleep(1)


if not os.path.exists('data'):
    os.makedirs('data')

# enter batch number(start from 1 and go on as you wish)
n = int(input('Enter the batch number: '))
file_name = 'data\\training_data_{}.npy'.format(n)

if os.path.isfile(file_name):
    print('File exists, loading previous data!')
    training_data = list(np.load(file_name, allow_pickle=True))
else:
    print('File does not exist, starting fresh!')
    training_data = []

main()
