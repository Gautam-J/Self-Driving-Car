import numpy as np
import pandas as pd
from collections import Counter
import os

'''
Run this script to check your data. The script will print the total number of
data along with the number of each label's occurence.
'''

n = int(input('Enter the batch number, 0 for final data: '))

if n == 0:
    if os.path.isfile('final_data.npy'):
        train_data = np.load('final_data.npy', allow_pickle=True)
        print(f'Total amount of frames collected: {len(train_data)}')

        df = pd.DataFrame(train_data)
        print(df.head())
        print('\n')
        print(Counter(df[2].apply(str)))
    else:
        print('Final data does not exist.')

else:
    train_data = 'data\\training_data_{}.npy'.format(n)
    if os.path.isfile(train_data):
        train_data_2 = np.load(train_data, allow_pickle=True)

        print('Unbalanced Raw Data: ' + str(len(train_data_2)))
        df = pd.DataFrame(train_data_2)

        print(df.head())
        print('\n')
        print(Counter(df[2].apply(str)))

        print('\n')

        train_data_bal = 'data\\training_data_{}_balanced.npy'.format(n)

        if os.path.isfile(train_data_bal):
            train_data_3 = np.load(train_data_bal, allow_pickle=True)

            print('New Balanced Data: ' + str(len(train_data_3)))
            df = pd.DataFrame(train_data_3)

            print(df.head())
            print('\n')
            print(Counter(df[2].apply(str)))

        else:
            print('Balanced data file does not exist.')

    else:
        print('Data does not exist.')
