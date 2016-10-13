__author__ = 'DaniloAbides'

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
import pandas as pd
from NN import baseline_model
import numpy as np

thinning = pd.read_csv('thinning_data.csv')
regular_data = pd.read_csv('train.csv')

X_train = (thinning.ix[:31999,1:].values).astype('float32')
X_test = (thinning.ix[32000:,1:].values).astype('float32')

y_train = (regular_data.ix[:31999,0].values).astype('int32')
y_test = (regular_data.ix[32000:,0].values).astype('int32')

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


num_pixels = X_train.shape[1]


model = baseline_model()

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=10, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

