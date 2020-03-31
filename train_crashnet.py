import os
import time
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from myhistory import MyHistory
from crashnet import CrashNet

BASE_DIR = f'models\\CrashNet\\{int(time.time())}'
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

data = np.load('data\\road_data.npy', allow_pickle=True).astype(np.float32)
print(data.shape)
data *= 1 / 255.

x_train, x_test = train_test_split(data, test_size=0.2, random_state=42)
print(x_train.shape)
print(x_test.shape)

_, _, model = CrashNet()
model.compile(loss='mse', optmizer=Adam(lr=1e-3, decay=1e-4))

es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
mh = MyHistory(model_name=f'{BASE_DIR}\\', win_size=32)

model.fit(x_train, x_train, epochs=2, batch_size=32, verbose=1,
          validation_split=0.2, callbacks=[es, mh])

model.save(f'{BASE_DIR}\\model.h5')
model.save_weights(f'{BASE_DIR}\\weights.h5')

json_config = model.to_json(indent=4)
with open(f'{BASE_DIR}\\model_config.json', 'w') as f:
    f.write(json_config)

print('Model saved.')
loss = model.evaluate(x_test, x_test, verbose=0)
print('Results on test set...')
print(f'Loss: {loss:0.5f}')

os.rename(BASE_DIR, f'{BASE_DIR}_{loss:0.5f}')
print('DONE')
