import cv2
import numpy as np
from grabscreen import grab_screen

'''
This script allows you to visualize the main screen getting captured by cv2.
The region shown by this script is the input data for the CNN.
Make sure that your game window is seen by cv2.
'''


def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked


while True:
    # change the region=(x, y, width, height) according to your game window.
    org_image = grab_screen(region=(270, 250, 650, 450))
    image = cv2.cvtColor(org_image, cv2.COLOR_BGR2GRAY)
    vertices = np.array([[0, 201], [0, 50], [381, 50], [381, 201]], np.int32)
    image = roi(image, [vertices])

    # org_image = cv2.resize(org_image, (200, 80))
    screen = cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB)
    cv2.imshow('window', screen)
    cv2.imshow('Region_of_interest', image)

    if cv2.waitKey(25) == ord('q'):
        cv2.destroyAllWindows()
        break
