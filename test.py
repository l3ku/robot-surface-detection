import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from extract_features import *
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers.core import Dense, Flatten
from keras.utils import to_categorical
from keras.applications.vgg16 import VGG16

if __name__ == '__main__':
    train_data = np.load('X_train_kaggle.npy')
    # all_id_classes = pd.read_csv('y_train_final_kaggle.csv')
    all_id_classes = np.genfromtxt('y_train_final_kaggle.csv', delimiter=',', dtype='str')
    # groups_csv = pd.read_csv('groups.csv').values
    groups_csv = np.genfromtxt('groups.csv', delimiter=',', dtype='str')
    le = preprocessing.LabelEncoder()
    le.fit(all_id_classes[:, 1])
    all_id_classes_transformed = le.transform(all_id_classes[:, 1])
    classes_array = np.array(all_id_classes_transformed)

    #train_data=train_data.reshape((1703,10,128,1))
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
    print(train_data.shape)

    train_data=np.pad(train_data,((0,0),(58,59),(0,0)),'wrap')
    t_data=np.stack((train_data,)*3,axis=-1)

    print(t_data.shape)

    #plt.matshow(train_data[1582])

    #plt.matshow(train_data[21])

    #plt.matshow(train_data[121])
    #plt.show()

    X_train,X_test,y_train,y_test=train_test_split(t_data, classes_array, test_size=0.2)

    print(X_test.shape)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    base_model=VGG16(include_top=False, weights='imagenet',input_shape=(127,127,3))
    w=base_model.output
    w=Flatten()(w)
    w=Dense(100,activation='relu')(w)
    output=Dense(9,activation='sigmoid')(w)

    model=Model(inputs=[base_model.input],outputs=[output])


    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='sgd',metrics = ['accuracy'])

    model.fit(X_train, y_train, batch_size=32, epochs = 5, validation_data = (X_test, y_test))

