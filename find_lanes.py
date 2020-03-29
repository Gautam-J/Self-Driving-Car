import cv2
import time
import numpy as np
from grabscreen import grab_screen
from directkeys import PressKey, ReleaseKey
from directkeys import W, A, D
from countdown import CountDown

'''
Most of the code in this script was taken from Sentdex's Python plays GTA-V
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


def auto_canny(image, sigma=0.33):
    '''
    Reference: https://www.pyimagesearch.com/
    '''
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged


def draw_lanes(img, lines, color=[0, 255, 255], thickness=3):

    # if this fails, go with some default line
    try:

        # finds the maximum y value for a lane marker
        # (since we cannot assume the horizon will always be at the same point.)

        ys = []
        for i in lines:
            for ii in i:
                ys += [ii[1], ii[3]]
        min_y = min(ys)
        max_y = 150
        new_lines = []
        line_dict = {}

        for idx, i in enumerate(lines):
            for xyxy in i:
                # These four lines:
                # modified from http://stackoverflow.com/questions/21565994/method-to-return-the-equation-of-a-straight-line-given-two-points
                # Used to calculate the definition of a line, given two sets of coords.
                x_coords = (xyxy[0], xyxy[2])
                y_coords = (xyxy[1], xyxy[3])
                A = np.vstack([x_coords, np.ones(len(x_coords))]).T
                m, b = np.linalg.lstsq(A, y_coords)[0]

                # Calculating our new, and improved, xs
                x1 = (min_y - b) / m
                x2 = (max_y - b) / m

                line_dict[idx] = [m, b, [int(x1), min_y, int(x2), max_y]]
                new_lines.append([int(x1), min_y, int(x2), max_y])

        final_lanes = {}

        for idx in line_dict:
            final_lanes_copy = final_lanes.copy()
            m = line_dict[idx][0]
            b = line_dict[idx][1]
            line = line_dict[idx][2]

            if len(final_lanes) == 0:
                final_lanes[m] = [[m, b, line]]

            else:
                found_copy = False

                for other_ms in final_lanes_copy:

                    if not found_copy:
                        if abs(other_ms * 1.2) > abs(m) > abs(other_ms * 0.8):
                            if abs(final_lanes_copy[other_ms][0][1] * 1.2) > abs(b) > abs(final_lanes_copy[other_ms][0][1] * 0.8):
                                final_lanes[other_ms].append([m, b, line])
                                found_copy = True
                                break
                        else:
                            final_lanes[m] = [[m, b, line]]

        line_counter = {}

        for lanes in final_lanes:
            line_counter[lanes] = len(final_lanes[lanes])

        top_lanes = sorted(line_counter.items(), key=lambda item: item[1])[::-1][:2]

        lane1_id = top_lanes[0][0]
        lane2_id = top_lanes[1][0]

        def average_lane(lane_data):
            x1s = []
            y1s = []
            x2s = []
            y2s = []
            for data in lane_data:
                x1s.append(data[2][0])
                y1s.append(data[2][1])
                x2s.append(data[2][2])
                y2s.append(data[2][3])
            return int(np.mean(x1s)), int(np.mean(y1s)), int(np.mean(x2s)), int(np.mean(y2s))

        l1_x1, l1_y1, l1_x2, l1_y2 = average_lane(final_lanes[lane1_id])
        l2_x1, l2_y1, l2_x2, l2_y2 = average_lane(final_lanes[lane2_id])

        return [l1_x1, l1_y1, l1_x2, l1_y2], [l2_x1, l2_y1, l2_x2, l2_y2], lane1_id, lane2_id
    except Exception:
        pass


def preprocess_img(image):
    org_image = image
    # convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gaussian blur
    image = cv2.GaussianBlur(image, (3, 3), 0)
    # edge detection
    image = auto_canny(image)
    # probabilistic hough transform
    lines = cv2.HoughLinesP(image, rho=1, theta=(np.pi / 180),
                            threshold=5, minLineLength=50, maxLineGap=15)
    m1 = 0
    m2 = 0
    # drawing lines
    try:
        l1, l2, m1, m2 = draw_lanes(org_image, lines)
        cv2.line(org_image, (l1[0], l1[1]), (l1[2], l1[3]), [0, 255, 0], 3)
        cv2.line(org_image, (l2[0], l2[1]), (l2[2], l2[3]), [0, 255, 0], 3)
    except Exception:
        pass
    try:
        for coords in lines:
            coords = coords[0]
            try:
                cv2.line(image, (coords[0], coords[1]), (coords[2], coords[3]), [255, 0, 0], 3)
            except Exception:
                pass
    except Exception:
        pass

    return image, org_image, m1, m2


CountDown(5)
while True:
    screen = grab_screen(region=(270, 280, 650, 450))
    new_screen, original_image, m1, m2 = preprocess_img(screen)
    cv2.imshow('window', new_screen)
    cv2.imshow('window2', cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))

    if m1 < 0 and m2 < 0:
        right()
    elif m1 > 0 and m2 > 0:
        left()
    else:
        straight()

    if cv2.waitKey(25) == ord('q'):
        cv2.destroyAllWindows()
        break
