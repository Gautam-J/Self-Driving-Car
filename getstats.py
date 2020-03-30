import os
import json
import pandas as pd
import numpy as np
from collections import Counter

'''
Script is used to iterate over all training data, and provide some insights
The result of this script will be plotted as an image and a json file will
be saved with all stats.
'''


def GetStats(path, filename):
    """Returns key stats in data

    Args:
        path (str): Path to .npy file whose data stats are required
        filename (str): Filename that will be added to the json file

    Returns:
        stats (dict): Stats of the data file passed in path arg.
    Raises:
        ValueError: if unknown element is found in data
    """

    print(f'Getting stats for {filename}...')
    data = np.load(path, allow_pickle=True)
    df = pd.DataFrame(data)
    c = Counter(df[2].apply(str))

    for element, count in c.items():
        if element == '[1, 0, 0]':
            left_count = count
        elif element == '[0, 1, 0]':
            straight_count = count
        elif element == '[0, 0, 1]':
            right_count = count
        else:
            raise ValueError

    total_count = left_count + right_count + straight_count
    file_size_MB = round(os.path.getsize(path) * 10**-6, 2)

    stats = [left_count, right_count, straight_count, total_count, file_size_MB]

    stats = {
        'File Name': filename,
        'Left Count': left_count,
        'Right Count': right_count,
        'Straight Count': straight_count,
        'Total Count': total_count,
        'File Size(MB)': file_size_MB
    }

    print(f'Got stats for {filename}\n')

    return stats


def main():
    MAX_BATCH = int(input('Enter maximum batch number: '))
    data_dict = dict()
    data_list = list()

    total_raw_stats = {
        'File Name': 'Total Raw Data',
        'Left Count': 0,
        'Right Count': 0,
        'Straight Count': 0,
        'Total Count': 0,
        'File Size(MB)': 0
    }

    for i in range(1, MAX_BATCH + 1):
        raw_file_path = f'data\\training_data_{i}.npy'
        raw_file_name = f'batch_{i}'
        bal_file_path = f'data\\training_data_{i}_balanced.npy'
        bal_file_name = f'batch_{i}_balanced'

        raw_stats = GetStats(raw_file_path, raw_file_name)
        bal_stats = GetStats(bal_file_path, bal_file_name)

        total_raw_stats['Left Count'] += raw_stats['Left Count']
        total_raw_stats['Right Count'] += raw_stats['Right Count']
        total_raw_stats['Straight Count'] += raw_stats['Straight Count']
        total_raw_stats['Total Count'] += raw_stats['Total Count']
        total_raw_stats['File Size(MB)'] += raw_stats['File Size(MB)']

        data_list.append(raw_stats)
        data_list.append(bal_stats)

    final_data_stats = GetStats('data\\final_data.npy', 'Final Data')
    total_raw_stats['File Size(MB)'] = round(total_raw_stats['File Size(MB)'], 2)
    data_list.append(total_raw_stats)
    data_list.append(final_data_stats)

    data_dict['stats'] = data_list

    with open('data_stats.json', 'w') as f:
        json.dump(data_dict, f, indent=4)


if __name__ == '__main__':
    main()
