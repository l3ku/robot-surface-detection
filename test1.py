import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from extract_features import *
from sklearn import preprocessing
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator

np.random.seed(25)

train_data = np.load('X_train_kaggle.npy')
test_data = np.load('X_test_kaggle.npy')
# all_id_classes = pd.read_csv('y_train_final_kaggle.csv')
all_id_classes = np.genfromtxt('y_train_final_kaggle.csv', delimiter=',', dtype='str')
# groups_csv = pd.read_csv('groups.csv').values
groups_csv = np.genfromtxt('groups.csv', delimiter=',', dtype='str')
le = preprocessing.LabelEncoder()
le.fit(all_id_classes[:, 1])
all_id_classes_transformed = le.transform(all_id_classes[:, 1])
classes_array = np.array(all_id_classes_transformed)


line1=train_data[:,0,:]
a1=line1-np.min(line1)
line1=np.divide(a1,np.max(line1)+np.finfo(float).eps)
line2=train_data[:,1,:]
a2=line2-np.min(line2)
line2=np.divide(a2,np.max(line2)+np.finfo(float).eps)
line3=train_data[:,2,:]
a3=line3-np.min(line3)
line3=np.divide(a3,np.max(line3)+np.finfo(float).eps)
line4=train_data[:,3,:]
a4=line4-np.min(line4)
line4=np.divide(a4,np.max(line4)+np.finfo(float).eps)
line5=train_data[:,4,:]
a5=line5-np.min(line5)
line5=np.divide(a5,np.max(line5)+np.finfo(float).eps)
line6=train_data[:,5,:]
a6=line6-np.min(line6)
line6=np.divide(a6,np.max(line6)+np.finfo(float).eps)
line7=train_data[:,6,:]
a7=line7-np.min(line7)
line7=np.divide(a7,np.max(line7)+np.finfo(float).eps)
line8=train_data[:,7,:]
a8=line8-np.min(line8)
line8=np.divide(a8,np.max(line8)+np.finfo(float).eps)
line9=train_data[:,8,:]
a9=line9-np.min(line9)
line9=np.divide(a9,np.max(line9)+np.finfo(float).eps)
line10=train_data[:,9,:]
a10=line10-np.min(line10)
line10=np.divide(a10,np.max(line10)+np.finfo(float).eps)
train_data=np.stack((line1,line2,line3,line4,line5,line6,line7,line8,line9,line10),axis=1)
train_data=np.diff(train_data)

train_data=np.pad(train_data,((0,0),(58,59),(0,0)),'wrap')

train_data=train_data.reshape((1703,127,127,1))
#XX=np.stack((XX,)*3,axis=-1)

X_train,X_test,y_train,y_test=train_test_split(train_data, classes_array, test_size=0.2)

X_train=train_data
y_train=classes_array


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(127,127,1)))
model.add(Activation('relu'))
BatchNormalization(axis=-1)
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

BatchNormalization(axis=-1)
model.add(Conv2D(64,(3, 3)))
model.add(Activation('relu'))
BatchNormalization(axis=-1)
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
# Fully connected layer

BatchNormalization()
model.add(Dense(512))
model.add(Activation('relu'))
BatchNormalization()
model.add(Dropout(0.2))
model.add(Dense(9))

# model.add(Convolution2D(10,3,3, border_mode='same'))
# model.add(GlobalAveragePooling2D())
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

gen = ImageDataGenerator(width_shift_range=0.08)

test_gen = ImageDataGenerator()

#train_generator = gen.flow(X_train, y_train, batch_size=64)
#test_generator = test_gen.flow(X_test, y_test, batch_size=64)

#model.fit_generator(train_generator, steps_per_epoch=1000, epochs=5)
#batch_size=16, nb_epoch=5, validation_data=(X_test, y_test)
model.fit(X_train, y_train,batch_size=16, nb_epoch=5)

# for test data


line1=test_data[:,0,:]
a1=line1-np.min(line1)
line1=np.divide(a1,np.max(line1)+np.finfo(float).eps)
line2=test_data[:,1,:]
a2=line2-np.min(line2)
line2=np.divide(a2,np.max(line2)+np.finfo(float).eps)
line3=test_data[:,2,:]
a3=line3-np.min(line3)
line3=np.divide(a3,np.max(line3)+np.finfo(float).eps)
line4=test_data[:,3,:]
a4=line4-np.min(line4)
line4=np.divide(a4,np.max(line4)+np.finfo(float).eps)
line5=test_data[:,4,:]
a5=line5-np.min(line5)
line5=np.divide(a5,np.max(line5)+np.finfo(float).eps)
line6=test_data[:,5,:]
a6=line6-np.min(line6)
line6=np.divide(a6,np.max(line6)+np.finfo(float).eps)
line7=test_data[:,6,:]
a7=line7-np.min(line7)
line7=np.divide(a7,np.max(line7)+np.finfo(float).eps)
line8=test_data[:,7,:]
a8=line8-np.min(line8)
line8=np.divide(a8,np.max(line8)+np.finfo(float).eps)
line9=test_data[:,8,:]
a9=line9-np.min(line9)
line9=np.divide(a9,np.max(line9)+np.finfo(float).eps)
line10=test_data[:,9,:]
a10=line10-np.min(line10)
line10=np.divide(a10,np.max(line10)+np.finfo(float).eps)
test_data=np.stack((line1,line2,line3,line4,line5,line6,line7,line8,line9,line10),axis=1)
test_data=np.diff(test_data)

test_data=np.pad(test_data,((0,0),(58,59),(0,0)),'wrap')

test_data=test_data.reshape((1705,127,127,1))

predicted = model.predict(test_data)

predicted=predicted.argmax(1)

print(predicted.shape)

labels = list(le.inverse_transform(predicted))

with open("submission.csv", "w") as fp:
	fp.write("# Id,Surface\n")

	for i, label in enumerate(labels):
		fp.write("%d,%s\n" % (i, label))
