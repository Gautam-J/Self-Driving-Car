import time


def CountDown(sec):
    digits = len(str(sec))
    delete = "\b" * (digits)
    for i in range(1, sec + 1)[::-1]:
        print("{0}{1:{2}}".format(delete, i, digits), end="", flush=True)
        time.sleep(1)
    print('')


if __name__ == '__main__':
    CountDown(5)
