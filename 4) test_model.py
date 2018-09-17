import numpy as np
import cv2
import time
from grabscreen import grab_screen
from getkeys import key_check
from alexnet import alexnet
from directkeys import PressKey, ReleaseKey, W, A, D

'''
Run this script to test your model in your game in real time. Make sure you
select your game window after initiating this script.
'''

MODEL_NAME = 'nfs-final.model'


def straight():
    print('straight')
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)


def left():
    print('left')
    PressKey(W)
    PressKey(A)
    time.sleep(0.09)
    ReleaseKey(A)


def right():
    print('right')
    PressKey(W)
    PressKey(D)
    time.sleep(0.09)
    ReleaseKey(D)


model = alexnet(width=86, height=56, output=3, channel=1, lr=0.001)
model.load(MODEL_NAME)
print('Model Loaded!')


def main():
    for i in list(range(5))[::-1]:
        print(i + 1)
        time.sleep(1)

    paused = False
    while True:

        if not paused:
            screen = grab_screen(region=(40, 250, 860, 560))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (86, 56))

            prediction = model.predict([screen.reshape(86, 56, 1)])[0]
            prediction = np.array(prediction) * [0.009, 5, 0.009]

            if np.argmax(prediction) == 0:
                left()
            elif np.argmax(prediction) == 1:
                straight()
            elif np.argmax(prediction) == 2:
                right()

        keys = key_check()

        if 'E' in keys:
            if paused:
                paused = False
                print('Unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)


main()
