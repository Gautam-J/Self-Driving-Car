import numpy as np
import cv2
import time
from grabscreen import grab_screen
from getkeys import key_check
from countdown import CountDown
from directkeys import PressKey, ReleaseKey, W, A, D
from tensorflow.keras.models import load_model

'''
Run this script to test your model in your game in real time. Make sure you
select your game window after initiating this script.
'''


def straight():
    print('straight')
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)


def left():
    print('left')
    PressKey(W)
    PressKey(A)
    time.sleep(0.05)
    ReleaseKey(A)


def right():
    print('right')
    PressKey(W)
    PressKey(D)
    time.sleep(0.05)
    ReleaseKey(D)


def main():
    CountDown(5)

    paused = False
    while True:

        if not paused:
            screen = grab_screen(region=(270, 250, 650, 450))
            minimap = grab_screen(region=(100, 390, 230, 490))

            screen = cv2.resize(screen, (200, 80))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            screen = screen.reshape(1, 80, 200, 3).astype(np.float32)

            minimap = cv2.resize(minimap, (50, 50))
            minimap = cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)
            minimap = minimap.reshape(1, 50, 50, 1).astype(np.float32)

            screen *= 1 / 255.
            minimap *= 1 / 255.

            prediction = model.predict([screen, minimap])[0]
            print(prediction)
            # prediction = np.array(prediction) * [0.009, 5, 0.009]

            if np.argmax(prediction) == 0:
                left()
            elif np.argmax(prediction) == 1:
                straight()
            elif np.argmax(prediction) == 2:
                right()

            # if prediction[0] >= THRESHOLD:
            #     left()
            # elif prediction[2] >= THRESHOLD:
            #     right()
            # else:
            #     straight()

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


MODEL_PATH = 'models\\1585417347_0.200_0.929\\model.h5'
THRESHOLD = 0.95

model = load_model(MODEL_PATH)
print('Model Loaded!')

if __name__ == '__main__':
    main()
