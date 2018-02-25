import numpy as np
import pandas as pd
from collections import Counter

train_data_2 = np.load('training_data.npy')
print('Unbalanced Raw Data: ' + str(len(train_data_2)))
df = pd.DataFrame(train_data_2)
print(df.head())
print(Counter(df[1].apply(str)))

print('\n')

train_data_3 = np.load('training_data_v2.npy')
print('New Balanced Data: ' + str(len(train_data_3)))
df = pd.DataFrame(train_data_3)
print(df.head())
print(Counter(df[1].apply(str)))
