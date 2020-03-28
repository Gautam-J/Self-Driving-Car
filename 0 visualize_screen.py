import cv2
from grabscreen import grab_screen

'''
This script allows you to visualize the main screen getting captured by cv2.
The region shown by this script is the input data for the CNN.
Make sure that your game window is seen by cv2.
'''

while True:
    # change the region=(x, y, width, height) according to your game window.
    screen = grab_screen(region=(270, 250, 650, 450))

    # uncomment the next line to see the resized image to be inputed.
    screen = cv2.resize(screen, (200, 80))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    cv2.imshow('window', screen)

    if cv2.waitKey(25) == ord('q'):
        cv2.destroyAllWindows()
        break
