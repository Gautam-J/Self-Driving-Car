import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle

'''
The problem with our data is that the occurence of going forward is way more
than that of going left, or right. If we feed this to a CNN, it would just
learn to go forward, no matter what. Therefore, we need to make the make the
number of instances of all labels equal. This is called balancing the data.

Note that you will lose a lot of your training data, but those are all the
unnecessary ones, that would ruin our CNN. Also, you have to run this script
for all your training batches.
'''

# enter 1, then run the script again and enter 2, and keep incrementing it.
max_batch = int(input("Enter the maximum batch number: "))

for n in range(1, max_batch + 1):
    train_data = np.load('data\\training_data_{}.npy'.format(n), allow_pickle=True)

    print(f'Frames collected for batch {n}: {len(train_data)}')
    df = pd.DataFrame(train_data)
    print(df.head())
    print('\n')
    print(Counter(df[2].apply(str)))
    print('\n')

    lefts = []
    rights = []
    forwards = []

    for data in train_data:
        screen = data[0]
        minimap = data[1]
        choice = data[2]

        if choice == [1, 0, 0]:
            lefts.append([screen, minimap, choice])
        elif choice == [0, 1, 0]:
            forwards.append([screen, minimap, choice])
        elif choice == [0, 0, 1]:
            rights.append([screen, minimap, choice])
        else:
            print('no matches!!!')

    if len(lefts) > len(rights):
        forwards = forwards[:len(rights)]
    else:
        forwards = forwards[:len(lefts)]

    lefts = lefts[:len(forwards)]
    rights = rights[:len(forwards)]

    final_data = forwards + lefts + rights
    shuffle(final_data)

    print(f'Balanced Data for batch {n}: {len(final_data)}')

    df = pd.DataFrame(final_data)
    print(df.head())
    print('\n')
    print(Counter(df[2].apply(str)))

    np.save('data\\training_data_{}_balanced.npy'.format(n), final_data)
    print('\n')
    print(f'Data Balanced and Saved for batch {n}!')
