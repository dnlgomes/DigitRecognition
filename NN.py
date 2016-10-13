__author__ = 'DaniloAbides'
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
import pandas as pd

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()





#### Sem features

# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255



# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, init='normal', activation='relu'))
    model.add(Dense(num_classes, init='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# build the model
model = baseline_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=10, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))



##### Sem feature

data = pd.read_csv("train.csv")
X_train = (data.ix[:31999,1:].values).astype('float32')
X_test = (data.ix[32000:,1:].values).astype('float32')

X_train = X_train / 255
X_test = X_test / 255

y_train = (data.ix[:31999,0].values).astype('int32')
y_test = (data.ix[32000:,0].values).astype('int32')

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

num_pixels = X_train.shape[1]

model = baseline_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=10, batch_size=200, verbose=2)

#### projecao vertical

proj_vertical = pd.read_csv('proj_vertical.csv', header=None)
max_value = max(proj_vertical.max())

X_train_proj_vertical = (proj_vertical.ix[:31999,:].values).astype('float32')
X_test_proj_vertical = (proj_vertical.ix[32000:,:].values).astype('float32')

X_train_proj_vertical = X_train_proj_vertical / max_value
X_test_proj_vertical = X_test_proj_vertical / max_value

X_train = np.column_stack((X_train,X_train_proj_vertical))
X_test = np.column_stack((X_test,X_test_proj_vertical))

num_pixels = X_train_proj_vertical.shape[1]

model = baseline_model()
# Fit the model
model.fit(X_train_proj_vertical, y_train, validation_data=(X_test_proj_vertical, y_test), nb_epoch=50, batch_size=200, verbose=2)



#### projecao horizontal

proj_horizontal = pd.read_csv('proj_horizontal.csv', header=None)
max_value = max(proj_horizontal.max())

X_train_proj_horizontal = (proj_horizontal.ix[:31999,:].values).astype('float32')
X_test_proj_horizontal = (proj_horizontal.ix[32000:,:].values).astype('float32')

X_train_proj_horizontal = X_train_proj_horizontal / max_value
X_test_proj_horizontal = X_test_proj_horizontal/ max_value



num_pixels = X_train_proj_horizontal.shape[1]

model = baseline_model()
# Fit the model
model.fit(X_train_proj_horizontal, y_train, validation_data=(X_test_proj_horizontal, y_test), nb_epoch=10, batch_size=200, verbose=2)




######### projecao horizontal + mlp normal

X_train_H_normal = np.column_stack((X_train,X_train_proj_horizontal))
X_test_H_normal = np.column_stack((X_test,X_test_proj_horizontal))

num_pixels = X_train_H_normal.shape[1]

model = baseline_model()
# Fit the model
model.fit(X_train_H_normal, y_train, validation_data=(X_test_H_normal, y_test), nb_epoch=50, batch_size=200, verbose=2)


######## proj vertical + horizontal


X_train_V_H = np.column_stack((X_train_proj_vertical,X_train_proj_horizontal, X_train))
X_test_V_H = np.column_stack((X_test_proj_vertical,X_test_proj_horizontal, X_test))

num_pixels = X_train_V_H.shape[1]

model = baseline_model()
# Fit the model
model.fit(X_train_V_H, y_train, validation_data=(X_test_V_H, y_test), nb_epoch=50, batch_size=200, verbose=2)
