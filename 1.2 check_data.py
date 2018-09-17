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

    train_data = 'final_data.npy'.format(n)

    if os.path.isfile(train_data):
        train_data_2 = np.load(train_data)

        print('Final Data: ' + str(len(train_data_2)))
        df = pd.DataFrame(train_data_2)

        print(df.head())
        print('\n')
        print(Counter(df[1].apply(str)))
    else:
        print('Final data does not exist.')

else:

    train_data = 'training_data_{}.npy'.format(n)

    if os.path.isfile(train_data):
        train_data_2 = np.load(train_data)

        print('Unbalanced Raw Data: ' + str(len(train_data_2)))
        df = pd.DataFrame(train_data_2)

        print(df.head())
        print('\n')
        print(Counter(df[1].apply(str)))

        print('\n')

        train_data_bal = 'training_data_{}_balanced.npy'.format(n)

        if os.path.isfile(train_data_bal):
            train_data_3 = np.load(train_data_bal)

            print('New Balanced Data: ' + str(len(train_data_3)))
            df = pd.DataFrame(train_data_3)

            print(df.head())
            print('\n')
            print(Counter(df[1].apply(str)))

        else:
            print('Balanced data file does not exist.')

    else:
        print('Data does not exist.')
