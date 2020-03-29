import os
import time
import numpy as np
from drivenet import DriveNet
from myhistory import MyHistory
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

BASE_DIR = f'models\\DriveNet\\{int(time.time())}'
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

data = np.load('data\\final_data.npy', allow_pickle=True)

screen, minimap, choice = [], [], []

for i in data:
    screen.append(i[0])
    minimap.append(i[1])
    choice.append(i[2])

data = train_test_split(screen, minimap, choice, test_size=0.2, random_state=42)
screen_train, screen_test, minimap_train, minimap_test, y_train, y_test = data

screen_train = np.array(screen_train, dtype=np.float32)
screen_test = np.array(screen_test, dtype=np.float32)
minimap_train = np.array(minimap_train, dtype=np.float32).reshape(-1, 50, 50, 1)
minimap_test = np.array(minimap_test, dtype=np.float32).reshape(-1, 50, 50, 1)
y_train = np.array(y_train, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)

screen_train *= 1 / 255.
screen_test *= 1 / 255.
minimap_train *= 1 / 255.
minimap_test *= 1 / 255.

# print(screen_train.shape)
# print(screen_test.shape)
# print(minimap_train.shape)
# print(minimap_test.shape)
# print(y_train.shape)
# print(y_test.shape)

model = DriveNet()
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
mh = MyHistory(model_name=f'{BASE_DIR}\\', win_size=32)

model.fit(x=[screen_train, minimap_train], y=y_train,
          epochs=5, batch_size=32, verbose=1, callbacks=[es, mh],
          validation_split=0.2)

model.save(f'{BASE_DIR}\\model.h5')
model.save_weights(f'{BASE_DIR}\\weights.h5')

json_config = model.to_json(indent=4)
with open(f'{BASE_DIR}\\model_config.json', 'w') as f:
    f.write(json_config)

print('Model saved.')
loss, acc = model.evaluate([screen_test, minimap_test], y_test, verbose=0)
print('Results on test set...')
print(f'Loss: {loss:0.3f} Accuracy: {acc:0.3f}')

os.rename(BASE_DIR, f'{BASE_DIR}_{loss:0.3f}_{acc:0.3f}')
print('DONE')
