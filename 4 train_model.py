import numpy as np
from alexnet import alexnet
from sklearn.model_selection import train_test_split

'''
We now feed our training data into a CNN. We use 'Alexnet' as our model.
We use a for loop for specifing epochs as we want to save the model after
every epoch.
'''

MODEL_NAME = 'nfs-final.model'

model = alexnet(height=56, width=86, lr=0.001, channel=1, output=3)
train = np.load('final_data.npy')

X = np.array([i[0] for i in train]).reshape(-1, 86, 56, 1)
Y = [i[1] for i in train]

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.1,
                                                    random_state=42)

for i in range(15):

    model.fit({'input': train_x}, {'targets': train_y}, n_epoch=1,
              validation_set=({'input': test_x}, {'targets': test_y}),
              snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

    model.save(MODEL_NAME)
    print('Saved epoch:', i+1)

# use the code in the next line in cmd to initiate tensorboard.
# tensorboard --logdir="log"
# or
# python -m tensorboard.main --logdir="log"
