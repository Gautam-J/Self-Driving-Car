import cv2
from grabscreen import grab_screen

'''
This script allows you to visualize the map getting captured by cv2.
The region shown by this script is the input data for the CNN.
Make sure that your game window is seen by cv2.
'''

while True:
    # change the region=(x, y, width, height) according to your game window.
    screen = grab_screen(region=(70, 360, 260, 520))

    # uncomment the next line to see the resized image to be inputed.
    screen = cv2.resize(screen, (50, 50))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    cv2.imshow('window', screen)

    if cv2.waitKey(25) == ord('q'):
        cv2.destroyAllWindows()
        break
