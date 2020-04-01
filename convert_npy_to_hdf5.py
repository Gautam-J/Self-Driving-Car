import os
import h5py
import numpy as np


def append_to_dataset(f, dataset, arr):
    f[dataset].resize(f[dataset].shape[0] + len(arr), axis=0)
    f[dataset][-len(arr):] = arr

    print(f'Shape of {dataset}: {f[dataset].shape}')


max_batch = int(input('Enter max batch number: '))

for i in range(1, max_batch + 1):
    data = np.load(f'data\\training_data_{i}.npy', allow_pickle=True)

    screen, minimap, choice = [], [], []
    for x, y, z in data:
        screen.append(x)
        minimap.append(y)
        choice.append(z)

    screen = np.array(screen)
    minimap = np.array(minimap).reshape(-1, 50, 50, 1)
    choice = np.array(choice)

    del data

    print(f'\nBatch {i}')
    print(screen.shape)
    print(minimap.shape)
    print(choice.shape)

    if not os.path.exists('data\\raw_data.hdf5'):

        with h5py.File('data\\raw_data.hdf5', 'w') as f:
            f.create_dataset('ScreenDataset', data=screen, dtype=np.float32,
                             maxshape=(None, 80, 200, 3), compression='lzf')

            f.create_dataset('MinimapDataset', data=minimap, dtype=np.float32,
                             maxshape=(None, 50, 50, 1), compression='lzf')

            f.create_dataset('ChoiceDataset', data=choice, dtype=np.float32,
                             maxshape=(None, 3), compression='lzf')
    else:

        with h5py.File('data\\raw_data.hdf5', 'a') as f:
            append_to_dataset(f, 'ScreenDataset', screen)
            append_to_dataset(f, 'MinimapDataset', minimap)
            append_to_dataset(f, 'ChoiceDataset', choice)

    del screen, minimap, choice
