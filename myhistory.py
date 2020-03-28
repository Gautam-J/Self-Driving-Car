import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from tensorflow.keras.callbacks import Callback

style.use('ggplot')

'''
Usage:

custom_history = MyHistory(model_name='directory\\name', win_size=64)

model.fit(x_train, y_train, epochs=20, batch_size=32,
          validation_split=0.2, ..., callbacks=[custom_history])
'''


class MyHistory(Callback):

    def __init__(self, model_name='', win_size=32):
        super(MyHistory, self).__init__()
        self.model_name = model_name
        self.win_size = win_size

    def on_train_begin(self, logs={}):
        self.batch_acc = []
        self.batch_loss = []
        self.epoch_acc = []
        self.epoch_loss = []
        self.val_acc = []
        self.val_loss = []

    def on_batch_end(self, batch, logs={}):
        self.batch_acc.append(logs.get('accuracy'))
        self.batch_loss.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_acc.append(logs.get('accuracy'))
        self.epoch_loss.append(logs.get('loss'))
        self.val_acc.append(logs.get('val_accuracy'))
        self.val_loss.append(logs.get('val_loss'))

    def on_train_end(self, logs={}):
        self.model.save(f'{self.model_name}-model_from_callback.h5')
        self.plot_training_history()

    def plot_training_history(self):
        plt.figure(figsize=(12, 7))

        plt.subplot(221)
        x = pd.DataFrame(self.batch_acc)
        x = x.rolling(window=self.win_size).mean().dropna()
        plt.plot(x, label='batch_acc')
        plt.xlabel('Batch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(222)
        plt.plot(self.epoch_acc, label='epoch_acc')
        plt.plot(self.val_acc, label='val_acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(223)
        x = pd.DataFrame(self.batch_loss)
        x = x.rolling(window=self.win_size).mean().dropna()
        plt.plot(x, label='batch_loss')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(224)
        plt.plot(self.epoch_loss, label='epoch_loss')
        plt.plot(self.val_loss, label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.savefig(f'{self.model_name}-history.png')
        plt.show()
