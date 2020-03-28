import time
from countdown import CountDown
from directkeys import PressKey, ReleaseKey
from directkeys import UP, DOWN, LEFT, RIGHT, ENTER, ESC, TWO


def left():
    PressKey(LEFT)
    time.sleep(0.1)
    ReleaseKey(LEFT)
    time.sleep(1)


def right():
    PressKey(RIGHT)
    time.sleep(0.1)
    ReleaseKey(RIGHT)
    time.sleep(1)


def up():
    PressKey(UP)
    time.sleep(0.1)
    ReleaseKey(UP)
    time.sleep(1)


def down():
    PressKey(DOWN)
    time.sleep(0.1)
    ReleaseKey(DOWN)
    time.sleep(1)


def enter():
    PressKey(ENTER)
    time.sleep(0.1)
    ReleaseKey(ENTER)
    time.sleep(1)


def esc():
    PressKey(ESC)
    time.sleep(0.1)
    ReleaseKey(ESC)
    time.sleep(1)


def two():
    PressKey(TWO)
    time.sleep(0.1)
    ReleaseKey(TWO)
    time.sleep(1)


free_roam = [enter, enter, enter, enter, enter, left, enter]
graphic = [esc, right, right, right, enter, right, enter, right, right, right, right, down,
           right, right, right, enter, left, enter]
advanced = [enter, two, up, right, up, right, up, right, enter, left, enter]


def StartGame():
    print('Make sure the game window is on focus...')
    CountDown(5)

    # going into free roam mode
    for action in free_roam:
        action()
    time.sleep(5)

    # changing graphic settings
    for action in graphic:
        action()
    time.sleep(5)

    # changing advanced graphic settings
    for action in advanced:
        action()

    esc()
    esc()

    print('Game is now ready')


if __name__ == '__main__':
    StartGame()
